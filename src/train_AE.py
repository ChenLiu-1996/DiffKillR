import argparse
import numpy as np
import torch
import os
import yaml
from model.scheduler import LinearWarmupCosineAnnealingLR
from tqdm import tqdm
from model.autoencoder import AutoEncoder
from utils.attribute_hashmap import AttributeHashmap
from utils.prepare_dataset import prepare_dataset
from utils.log_util import log
from utils.metrics import clustering_accuracy, topk_accuracy, embedding_mAP
from utils.parse import parse_settings
from utils.seed import seed_everything
from utils.early_stop import EarlyStopping
from loss.supervised_contrastive import SupConLoss
from loss.triplet_loss import TripletLoss

def construct_triplet_batch(img_paths,
                            latent_features,
                            num_pos,
                            num_neg,
                            model,
                            dataset,
                            split,
                            device):
    '''
        Returns:
        pos_features: (bsz * num_pos, latent_dim)
        neg_features: (bsz * num_neg, latent_dim)

    '''
    pos_images = None
    cell_type_labels = [] # (bsz)
    pos_cell_type_labels = [] # (bsz * num_pos)

    # Positive.
    for img_path in img_paths:
        cell_type = dataset.get_celltype(img_path=img_path)
        cell_type_labels.append(dataset.cell_type_to_idx[cell_type])
        pos_cell_type_labels.extend([dataset.cell_type_to_idx[cell_type]] * num_pos)

        aug_images, _ = dataset.sample_celltype(split=split,
                                                celltype=cell_type,
                                                cnt=num_pos)
        aug_images = torch.Tensor(aug_images).to(device)

        if pos_images is not None:
            pos_images = torch.cat([pos_images, aug_images], dim=0)
        else:
            pos_images = aug_images
    _, pos_features = model(pos_images) # (bsz * num_pos, latent_dim)

    # Negative.
    num_neg = config.num_neg
    neg_features = None # (bsz*num_neg, latent_dim)
    all_features = torch.cat([latent_features, pos_features], dim=0) # (bsz * (1+num_pos), latent_dim)

    all_cell_type_labels = cell_type_labels.copy()
    all_cell_type_labels.extend(pos_cell_type_labels) # (bsz * (1+num_pos))

    for img_path in img_paths:
        cell_type = dataset.get_celltype(img_path=img_path)

        negative_pool = np.argwhere(
            (np.array(all_cell_type_labels) != dataset.cell_type_to_idx[cell_type]) * 1).flatten()

        neg_idxs = np.random.choice(negative_pool, size=num_neg, replace=False)

        if neg_features is not None:
            neg_features = torch.cat([neg_features, all_features[neg_idxs]], dim=0)
        else:
            neg_features = all_features[neg_idxs]

    return pos_features, neg_features


def construct_batch_images_with_n_views(
        images,
        img_paths,
        dataset,
        n_views,
        sampling_method,
        split,
        device):
    '''
        Returns:
        batch_images: [bsz * n_views, ...],
        cell_type_labels: [bsz * n_views]
    '''
    if sampling_method not in ['supercontrast', 'SimCLR']:
        raise ValueError('`sampling_method`: %s not supported.' % sampling_method)
    
    # Construct batch_images [bsz * n_views, in_chan, H, W].
    batch_images = None
    cell_type_labels = []
    n_views = config.n_views
    for image, img_path in zip(images, img_paths):
        cell_type = dataset.get_celltype(img_path=img_path)
        cell_type_labels.append(dataset.cell_type_to_idx[cell_type])
        if n_views > 1:
            if sampling_method == 'supercontrast':
                aug_images, _ = dataset.sample_celltype(split=split,
                                                        celltype=cell_type,
                                                        cnt=n_views-1)
            elif sampling_method == 'SimCLR':
                patch_id = dataset.get_patch_id(img_path=img_path)
                aug_images, _ = dataset.sample_views(split=split,
                                                     patch_id=patch_id,
                                                     cnt=n_views-1)
                
            aug_images = torch.Tensor(aug_images).to(device) # (cnt, in_chan, H, W)

            image = torch.unsqueeze(image, dim=0) # (1, in_chan, H, W)
            if batch_images is not None:
                batch_images = torch.cat([batch_images, image, aug_images], dim=0)
            else:
                batch_images = torch.cat([image, aug_images], dim=0)
        else:
            if batch_images is not None:
                batch_images = torch.cat([batch_images, image], dim=0)
            else:
                batch_images = torch.cat([image], dim=0)

    batch_images = batch_images.float().to(device) #[bsz * n_views, in_chan, H, W].

    return batch_images, cell_type_labels




def train(config: AttributeHashmap):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    dataset, train_set, val_set, _ = \
        prepare_dataset(config=config)

    # Build the model
    try:
        model = globals()[config.model](num_filters=config.num_filters,
                                        in_channels=3,
                                        out_channels=3)
    except:
        raise ValueError('`config.model`: %s not supported.' % config.model)

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=10,
        warmup_start_lr=float(config.learning_rate) / 100,
        max_epochs=config.max_epochs,
        eta_min=0)

    if config.latent_loss in ['supercontrast', 'SimCLR']:
        supercontrast_loss = SupConLoss(temperature=config.temp,
                                        base_temperature=config.base_temp,
                                        contrast_mode=config.contrast_mode)
    elif config.latent_loss == 'triplet':
        triplet_loss = TripletLoss(distance_measure='cosine',
                                   margin=config.margin,
                                   num_pos=config.num_pos,
                                   num_neg=config.num_neg)
    mse_loss = torch.nn.MSELoss()
    early_stopper = EarlyStopping(mode='min',
                                  patience=config.patience,
                                  percentage=False)

    best_val_loss = np.inf

    for epoch_idx in tqdm(range(config.max_epochs)):
        train_loss, train_latent_loss, train_recon_loss = 0, 0, 0
        model.train()
        for iter_idx, (images, _, canonical_images, _, img_paths, _) in enumerate(tqdm(train_set)):
            images = images.float().to(device) # (bsz, in_chan, H, W)
            canonical_images = canonical_images.float().to(device)
            bsz = images.shape[0]

            '''
                Reconstruction loss.
            '''
            recon_images, latent_features = model(images)
            recon_loss = mse_loss(recon_images, canonical_images)


            '''
                Latent embedding loss.
            '''
            latent_loss = None
            if config.latent_loss == 'supercontrast' or config.latent_loss == 'SimCLR':
                sampling_method = config.latent_loss
                batch_images, cell_type_labels = construct_batch_images_with_n_views(images,
                                                                                    img_paths,
                                                                                    dataset,
                                                                                    config.n_views,
                                                                                    sampling_method,
                                                                                    'train',
                                                                                    device)
                _, latent_features = model(batch_images) # (bsz * n_views, latent_dim)
                latent_features = latent_features.contiguous().view(bsz, config.n_views, -1) # (bsz, n_views, latent_dim)
                cell_type_labels = torch.tensor(cell_type_labels).to(device) # (bsz)

                if sampling_method == 'supercontrast':
                    latent_loss = supercontrast_loss(features=latent_features,
                                                        labels=cell_type_labels)
                else:
                    # Both `labels` and `mask` are None, perform SimCLR unsupervised loss:
                    latent_loss = supercontrast_loss(features=latent_features)

            elif config.latent_loss == 'triplet':
                pos_features, neg_features = construct_triplet_batch(img_paths,
                                                                    latent_features,
                                                                    config.num_pos,
                                                                    config.num_neg,
                                                                    model,
                                                                    dataset,
                                                                    'train',
                                                                    device=device)
                pos_features = pos_features.contiguous().view(bsz, config.num_pos, -1) # (bsz, num_pos, latent_dim)
                neg_features = neg_features.contiguous().view(bsz, config.num_neg, -1) # (bsz, num_neg, latent_dim)

                latent_loss = triplet_loss(anchor=latent_features,
                                           positive=pos_features,
                                           negative=neg_features)
            else:
                raise ValueError('`config.latent_loss`: %s not supported.' % config.latent_loss)


            loss = latent_loss + recon_loss
            train_loss += loss.item()
            train_latent_loss += latent_loss.item()
            train_recon_loss += recon_loss.item()
            # print('\rIter %d, loss: %.3f, contrastive: %.3f, recon: %.3f\n' % (
            #     iter_idx, loss.item(), latent_loss.item(), recon_loss.item()
            # ))

            # Simulate `config.batch_size` by batched optimizer update.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = train_loss / (iter_idx + 1) # avg loss over minibatches
        train_latent_loss = train_latent_loss / (iter_idx + 1)
        train_recon_loss = train_recon_loss / (iter_idx + 1)

        lr_scheduler.step()

        log('Train [%s/%s] loss: %.3f, latent: %.3f, recon: %.3f'
            % (epoch_idx + 1, config.max_epochs, train_loss, train_latent_loss,
               train_recon_loss),
            filepath=config.log_dir,
            to_console=False)

        # Validation.
        model.eval()
        with torch.no_grad():
            val_loss, val_latent_loss, val_recon_loss = 0, 0, 0
            for iter_idx, (images, _, canonical_images, _, img_paths, _) in enumerate(tqdm(val_set)):
                # NOTE: batch size is len(val_set) here.
                # May need to change this if val_set is too large.
                images = images.float().to(device)
                canonical_images = canonical_images.float().to(device)
                bsz = images.shape[0]

                recon_images, latent_features = model(images)
                recon_loss = mse_loss(recon_images, canonical_images)

                latent_loss = None
                if config.latent_loss == 'supercontrast' or config.latent_loss == 'SimCLR':
                    sampling_method = config.latent_loss
                    batch_images, cell_type_labels = construct_batch_images_with_n_views(images,
                                                                                        img_paths,
                                                                                        dataset,
                                                                                        config.n_views,
                                                                                        sampling_method,
                                                                                        'val', # NOTE: use val
                                                                                        device)
                    _, latent_features = model(batch_images) # (bsz * n_views, latent_dim)
                    latent_features = latent_features.contiguous().view(bsz, config.n_views, -1) # (bsz, n_views, latent_dim)
                    cell_type_labels = torch.tensor(cell_type_labels).to(device) # (bsz)

                    if sampling_method == 'supercontrast':
                        latent_loss = supercontrast_loss(features=latent_features,
                                                            labels=cell_type_labels)
                    else:
                        # Both `labels` and `mask` are None, perform SimCLR unsupervised loss:
                        latent_loss = supercontrast_loss(features=latent_features)
                elif config.latent_loss == 'triplet':
                    pos_features, neg_features = construct_triplet_batch(img_paths,
                                                                        latent_features,
                                                                        config.num_pos,
                                                                        config.num_neg,
                                                                        model,
                                                                        dataset,
                                                                        'val',
                                                                        device=device)
                    pos_features = pos_features.contiguous().view(bsz, config.num_pos, -1) # (bsz, num_pos, latent_dim)
                    neg_features = neg_features.contiguous().view(bsz, config.num_neg, -1) # (bsz, num_neg, latent_dim)

                    latent_loss = triplet_loss(anchor=latent_features,
                                            positive=pos_features,
                                            negative=neg_features)
                else:
                    raise ValueError('`config.latent_loss`: %s not supported.' % config.latent_loss)

                val_latent_loss += latent_loss.item()
                val_recon_loss += recon_loss.item()
                val_loss += val_latent_loss + val_recon_loss

        val_loss /= (iter_idx + 1)
        val_latent_loss /= (iter_idx + 1)
        val_recon_loss /= (iter_idx + 1)

        log('Validation [%s/%s] loss: %.3f, latent: %.3f, recon: %.3f\n'
            % (epoch_idx + 1, config.max_epochs, val_loss, val_latent_loss,
               val_recon_loss),
            filepath=config.log_dir,
            to_console=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(config.model_save_path)
            log('%s: Model weights successfully saved.' % config.model,
                filepath=config.log_dir,
                to_console=False)

        if early_stopper.step(val_loss):
            log('Early stopping criterion met. Ending training.',
                filepath=config.log_dir,
                to_console=True)
            break

    return


@torch.no_grad()
def test(config: AttributeHashmap):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    dataset, train_set, val_set, test_set = prepare_dataset(config=config)

    # Build the model
    try:
        model = globals()[config.model](num_filters=config.num_filters,
                                        in_channels=3,
                                        out_channels=3)
    except:
        raise ValueError('`config.model`: %s not supported.' % config.model)

    model = model.to(device)
    model.load_weights(config.model_save_path, device=device)
    log('%s: Model weights successfully loaded.' % config.model,
        to_console=True)

    save_path_fig_embeddings = '%s/results/embeddings.png' % config.output_save_path
    save_path_fig_reconstructed = '%s/results/reconstructed.png' % config.output_save_path
    os.makedirs(os.path.dirname(save_path_fig_embeddings), exist_ok=True)

    if config.latent_loss in ['supercontrast', 'SimCLR']:
        supercontrast_loss = SupConLoss(temperature=config.temp,
                                        base_temperature=config.base_temp,
                                        contrast_mode=config.contrast_mode)
    elif config.latent_loss == 'triplet':
        triplet_loss = TripletLoss(distance_measure='cosine',
                                   margin=config.margin,
                                   num_pos=config.num_pos,
                                   num_neg=config.num_neg)
    else:
        raise ValueError('`config.latent_loss`: %s not supported.' % config.latent_loss)

    mse_loss = torch.nn.MSELoss()

    # Test.
    model.eval()
    with torch.no_grad():
        test_loss, test_latent_loss, test_recon_loss = 0, 0, 0
        for iter_idx, (images, _, canonical_images, _, img_paths, _) in enumerate(tqdm(test_set)):
            images = images.float().to(device)
            canonical_images = canonical_images.float().to(device)
            bsz = images.shape[0]

            recon_images, latent_features = model(images)
            recon_loss = mse_loss(recon_images, canonical_images)

            latent_loss = None
            if config.latent_loss == 'supercontrast' or config.latent_loss == 'SimCLR':
                sampling_method = config.latent_loss
                batch_images, cell_type_labels = construct_batch_images_with_n_views(images,
                                                                                    img_paths,
                                                                                    dataset,
                                                                                    config.n_views,
                                                                                    sampling_method,
                                                                                    'test', # NOTE: use test
                                                                                    device)
                _, latent_features = model(batch_images) # (bsz * n_views, latent_dim)
                latent_features = latent_features.contiguous().view(bsz, config.n_views, -1) # (bsz, n_views, latent_dim)
                cell_type_labels = torch.tensor(cell_type_labels).to(device) # (bsz)

                if sampling_method == 'supercontrast':
                    latent_loss = supercontrast_loss(features=latent_features,
                                                        labels=cell_type_labels)
                else:
                    # Both `labels` and `mask` are None, perform SimCLR unsupervised loss:
                    latent_loss = supercontrast_loss(features=latent_features)
            elif config.latent_loss == 'triplet':
                pos_features, neg_features = construct_triplet_batch(img_paths,
                                                                    latent_features,
                                                                    config.num_pos,
                                                                    config.num_neg,
                                                                    model,
                                                                    dataset,
                                                                    'test',
                                                                    device=device)
                pos_features = pos_features.contiguous().view(bsz, config.num_pos, -1) # (bsz, num_pos, latent_dim)
                neg_features = neg_features.contiguous().view(bsz, config.num_neg, -1) # (bsz, num_neg, latent_dim)

                latent_loss = triplet_loss(anchor=latent_features,
                                           positive=pos_features,
                                           negative=neg_features)
            else:
                raise ValueError('`config.latent_loss`: %s not supported.' % config.latent_loss)

            test_latent_loss += latent_loss.item()
            test_recon_loss += recon_loss.item()
            test_loss += test_latent_loss + test_recon_loss

    test_loss /= (iter_idx + 1)
    test_latent_loss /= (iter_idx + 1)
    test_recon_loss /= (iter_idx + 1)

    log('Test loss: %.3f, contrastive: %.3f, recon: %.3f\n'
        % (test_loss, test_latent_loss, test_recon_loss),
        filepath=config.log_dir,
        to_console=False)


    # Visualize latent embeddings.
    embeddings = {split: None for split in ['train', 'val', 'test']}
    embedding_labels = {split: [] for split in ['train', 'val', 'test']}
    embedding_labels_int = {split: [] for split in ['train', 'val', 'test']}
    embedding_patch_id_int = {split: [] for split in ['train', 'val', 'test']}
    og_inputs = {split: None for split in ['train', 'val', 'test']}
    canonical = {split: None for split in ['train', 'val', 'test']}
    reconstructed = {split: None for split in ['train', 'val', 'test']}

    with torch.no_grad():
        for split, split_set in zip(['train', 'val', 'test'], [train_set, val_set, test_set]):
            for iter_idx, (images, _, canonical_images, _, img_paths, _) in enumerate(tqdm(split_set)):
                images = images.float().to(device)
                recon_images, latent_features = model(images)
                latent_features = torch.flatten(latent_features, start_dim=1)

                # Move to cpu to save memory on gpu.
                images = images.cpu()
                recon_images = recon_images.cpu()
                latent_features = latent_features.cpu()

                if embeddings[split] is None:
                    embeddings[split] = latent_features  # (bsz, latent_dim)
                else:
                    embeddings[split] = torch.cat([embeddings[split], latent_features], dim=0)
                if reconstructed[split] is None:
                    reconstructed[split] = recon_images # (bsz, in_chan, H, W)
                else:
                    reconstructed[split] = torch.cat([reconstructed[split], recon_images], dim=0)
                if og_inputs[split] is None:
                    og_inputs[split] = images
                else:
                    og_inputs[split] = torch.cat([og_inputs[split], images], dim=0)
                if canonical[split] is None:
                    canonical[split] = canonical_images
                else:
                    canonical[split] = torch.cat([canonical[split], canonical_images], dim=0)
                embedding_labels[split].extend([dataset.get_celltype(img_path=img_path) for img_path in img_paths])
                embedding_patch_id_int[split].extend([dataset.get_patch_id_idx(img_path=img_path) for img_path in img_paths])


            embeddings[split] = embeddings[split].numpy()
            reconstructed[split] = reconstructed[split].numpy()
            og_inputs[split] = og_inputs[split].numpy()
            canonical[split] = canonical[split].numpy()
            embedding_labels_int[split] = np.array([dataset.get_celltype_id(ct) for ct in embedding_labels[split]])
            embedding_labels[split] = np.array(embedding_labels[split])
            embedding_patch_id_int[split] = np.array(embedding_patch_id_int[split])
            assert len(embeddings[split]) == len(reconstructed[split]) == len(embedding_labels[split])
            assert len(embedding_labels[split]) == len(embedding_labels_int[split])

    # Quantify latent embedding quality.
    ins_clustering_acc, class_clustering_acc = {}, {}
    ins_topk_acc, ins_mAP = {}, {}
    class_topk_acc, class_mAP = {}, {}

    for split in ['train', 'val', 'test']:
        if config.latent_loss == 'triplet':
            distance_measure = 'cosine'
        elif config.latent_loss == 'supercontrast':
            distance_measure = 'cosine'
        elif config.latent_loss == 'SimCLR':
            distance_measure = 'cosine'

        class_adj = np.zeros((len(embedding_labels[split]), len(embedding_labels[split])))
        instance_adj = np.zeros((len(embedding_labels[split]), len(embedding_labels[split])))
        log(f'Constructing class ({class_adj.shape}) and \
            instance adjacency ({instance_adj.shape}) matrices...', to_console=True)
        for i in range(len(embedding_labels[split])):
            for j in range(len(embedding_labels[split])):
                if embedding_labels_int[split][i] == embedding_labels_int[split][j]:
                    class_adj[i, j] = 1
                
                # same patch id means same instance
                if embedding_patch_id_int[split][i] == embedding_patch_id_int[split][j]:
                    instance_adj[i, j] = 1
        log(f'Done constructing class ({class_adj.shape}) and \
            instance adjacency ({instance_adj.shape}) matrices.', to_console=True)

        ins_clustering_acc[split] = clustering_accuracy(embeddings[split],
                                                        embeddings['train'],
                                                        embedding_patch_id_int[split],
                                                        embedding_patch_id_int['train'],
                                                        distance_measure=distance_measure,
                                                        voting_k=1)
        class_clustering_acc[split] = clustering_accuracy(embeddings[split],
                                                        embeddings['train'],
                                                        embedding_labels_int[split],
                                                        embedding_labels_int['train'],
                                                        distance_measure=distance_measure,
                                                        voting_k=1)
        ins_topk_acc[split] = topk_accuracy(embeddings[split],
                                            instance_adj,
                                            distance_measure=distance_measure,
                                            k=1)
        ins_mAP[split] = embedding_mAP(embeddings[split],
                                       instance_adj,
                                       distance_op=distance_measure)

        class_topk_acc[split] = topk_accuracy(embeddings[split],
                                              class_adj,
                                              distance_measure=distance_measure,
                                              k=1)
        class_mAP[split] = embedding_mAP(embeddings[split],
                                         class_adj,
                                         distance_op=distance_measure)
        log(f'Instance clustering accuracy: {ins_clustering_acc[split]:.3f}', to_console=True)
        log(f'Class clustering accuracy: {class_clustering_acc[split]:.3f}', to_console=True)
        log(f'Instance top-k accuracy: {ins_topk_acc[split]:.3f}', to_console=True)
        log(f'Class top-k accuracy: {class_topk_acc[split]:.3f}', to_console=True)
        log(f'Instance mAP: {ins_mAP[split]:.3f}', to_console=True)
        log(f'Class mAP: {class_mAP[split]:.3f}', to_console=True)


    # Plot latent embeddings
    import phate
    import scprep
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'serif'

    fig_embedding = plt.figure(figsize=(10, 8 * 3))
    for split in ['train', 'val', 'test']:
        phate_op = phate.PHATE(random_state=0,
                                 n_jobs=1,
                                 n_components=2,
                                 knn_dist='cosine',
                                 verbose=False)
        data_phate = phate_op.fit_transform(embeddings[split])
        print('Visualizing ', split, ' : ',  data_phate.shape)
        ax = fig_embedding.add_subplot(3, 1, ['train', 'val', 'test'].index(split) + 1)
        title = f"{split}:Instance clustering acc: {ins_clustering_acc[split]:.3f},\n \
            Class clustering acc: {class_clustering_acc[split]:.3f},\n \
            Instance top-k acc: {ins_topk_acc[split]:.3f},\n \
            Class top-k acc: {class_topk_acc[split]:.3f},\n \
            Instance mAP: {ins_mAP[split]:.3f},\n \
            Class mAP: {class_mAP[split]:.3f}"
        
        scprep.plot.scatter2d(data_phate,
                              c=embedding_labels[split],
                              legend=dataset.cell_types,
                              ax=ax,
                              title=title,
                              xticks=False,
                              yticks=False,
                              label_prefix='PHATE',
                              fontsize=10,
                              s=5)
    plt.tight_layout()
    plt.savefig(save_path_fig_embeddings)
    plt.close(fig_embedding)


    # Visualize reconstruction
    sample_n = 5
    fig_reconstructed = plt.figure(figsize=(2 * sample_n * 3, 2 * 3))  # (W, H)
    fig_reconstructed.suptitle('Reconstruction of canonical view', fontsize=10)
    for split in ['train', 'val', 'test']:
        for i in range(sample_n):
            # Original Input
            ax = fig_reconstructed.add_subplot(3, sample_n * 3,
                                               ['train', 'val', 'test'].index(split) * sample_n * 3 + i * 3 + 1)
            sample_idx = np.random.randint(low=0, high=len(reconstructed[split]))
            img_display = og_inputs[split][sample_idx].transpose(1, 2, 0)  # (H, W, in_chan)
            img_display = np.clip((img_display + 1) / 2, 0, 1)
            ax.imshow(img_display)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_ylabel(split, fontsize=10)
            ax.set_title('Original')

            # Canonical
            ax = fig_reconstructed.add_subplot(3, sample_n * 3,
                                               ['train', 'val', 'test'].index(split) * sample_n * 3 + i * 3 + 2)
            img_display = canonical[split][sample_idx].transpose(1, 2, 0)  # (H, W, in_chan)
            img_display = np.clip((img_display + 1) / 2, 0, 1)
            ax.imshow(img_display)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_ylabel(split, fontsize=10)
            ax.set_title('Canonical')

            # Reconstructed
            ax = fig_reconstructed.add_subplot(3, sample_n * 3,
                                               ['train', 'val', 'test'].index(split) * sample_n * 3 + i * 3 + 3)
            img_display = reconstructed[split][sample_idx].transpose(1, 2, 0)  # (H, W, in_chan)
            img_display = np.clip((img_display + 1) / 2, 0, 1)
            ax.imshow(img_display)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('Reconstructed')

    plt.tight_layout()
    plt.savefig(save_path_fig_reconstructed)
    plt.close(fig_reconstructed)


    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--mode', help='`train` or `test`?', required=True)
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)
    parser.add_argument('--run_count', help='Provide this during testing!', default=1)
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument('--num-workers', help='Number of workers, e.g. use number of cores', default=4, type=int)
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config.gpu_id = args.gpu_id
    config.num_workers = args.num_workers
    config = parse_settings(config, log_settings=args.mode == 'train', run_count=args.run_count)

    assert args.mode in ['train', 'test']

    seed_everything(config.random_seed)

    if args.mode == 'train':
        train(config=config)
        test(config=config)
    elif args.mode == 'test':
        test(config=config)
