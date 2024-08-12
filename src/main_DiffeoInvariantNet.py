'''
Main file for DiffeoInvariantNet.
'''

import argparse
import numpy as np
import torch
import sklearn.metrics
import pandas as pd

import os
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader

import ast

from model.scheduler import LinearWarmupCosineAnnealingLR
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
from datasets.augmented_MoNuSeg import AugmentedMoNuSegDataset, load_image
from datasets.augmented_GLySAC import AugmentedGLySACDataset


def train(config, wandb_run=None):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')

    dataset, train_loader, val_loader, _ = prepare_dataset(config=config)

    # Build the model
    try:
        model = globals()[config.DiffeoInvariantNet_model](num_filters=config.num_filters,
                                                           depth=config.depth,
                                                           in_channels=3,
                                                           out_channels=3)
    except:
        raise ValueError('`config.DiffeoInvariantNet_model`: %s not supported.' % config.DiffeoInvariantNet_model)

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=10,
        warmup_start_lr=float(config.learning_rate) / 100,
        max_epochs=config.max_epochs,
        eta_min=0)

    if config.latent_loss in ['SimCLR']:
        supcontrast_loss = SupConLoss(temperature=config.temp,
                                      base_temperature=config.base_temp,
                                      contrastive_mode=config.contrastive_mode)
    elif config.latent_loss == 'triplet':
        triplet_loss = TripletLoss(distance_measure='cosine',
                                   margin=config.margin,
                                   num_pos=config.num_pos,
                                   num_neg=config.num_neg)
    else:
        raise ValueError('`config.latent_loss`: %s not supported.' % config.latent_loss)

    mse_loss = torch.nn.MSELoss()
    early_stopper = EarlyStopping(mode='min',
                                  patience=config.patience)

    best_val_loss = np.inf

    for epoch_idx in range(config.max_epochs):
        train_loss, train_latent_loss, train_recon_loss = 0, 0, 0
        model.train()

        for iter_idx, (images, _, image_n_view, _, canonical_images, _) in enumerate(train_loader):
            images = images.float().to(device)  # [batch_size, C, H, W]

            # [batch_size, n_views, C, H, W] -> [batch_size * n_views, C, H, W]
            image_n_view = image_n_view.reshape(-1, image_n_view.shape[-3], image_n_view.shape[-2], image_n_view.shape[-1])
            image_n_view = image_n_view.float().to(device)

            canonical_images = canonical_images.float().to(device)
            batch_size = images.shape[0]

            '''
            Reconstruction loss.
            '''
            recon_images, latent_features = model(images)
            recon_loss = mse_loss(recon_images, canonical_images)

            '''
            Latent embedding loss.
            '''

            latent_loss = None
            if config.latent_loss == 'supcontrast' or config.latent_loss == 'SimCLR':
                _, latent_features = model(image_n_view) # (batch_size * n_views, C_dim, H_dim, W_dim)
                latent_features = latent_features.contiguous().view(batch_size, config.n_views, -1) # (batch_size, n_views, latent_dim)

                # Both `labels` and `mask` are None, perform SimCLR unsupervised loss:
                latent_loss = supcontrast_loss(features=latent_features)

            elif config.latent_loss == 'triplet':

                _, anchor_features = model(images) # (batch_size, C_dim, H_dim, W_dim)
                _, other_features = model(image_n_view) # (batch_size * n_views, C_dim, H_dim, W_dim)

                anchor_features = anchor_features.contiguous().view(batch_size, -1) # (batch_size, latent_dim)
                pos_neg_features = other_features.contiguous().view(batch_size, config.num_pos + config.num_neg, -1)
                pos_features = pos_neg_features[:, :config.num_pos, :] # (batch_size, num_pos, latent_dim)
                neg_features = pos_neg_features[:, config.num_pos:, :] # (batch_size, num_neg, latent_dim)

                latent_loss = triplet_loss(anchor=anchor_features, positive=pos_features, negative=neg_features)

            else:
                raise ValueError('`config.latent_loss`: %s not supported.' % config.latent_loss)


            loss = latent_loss + recon_loss
            train_loss += loss.item()
            train_latent_loss += latent_loss.item()
            train_recon_loss += recon_loss.item()

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
            filepath=config.log_path,
            to_console=True)
        # if wandb_run is not None:
        #     wandb.log({'train/loss': train_loss,
        #                'train/latent_loss': train_latent_loss,
        #                'train/recon_loss': train_recon_loss})

        # Validation.
        model.eval()
        with torch.no_grad():
            val_loss, val_latent_loss, val_recon_loss = 0, 0, 0
            for iter_idx, (images, _, image_n_view, _, canonical_images, _) in enumerate(val_loader):
                images = images.float().to(device)  # [batch_size, C, H, W]

                # [batch_size, n_views, C, H, W] -> [batch_size * n_views, C, H, W]
                image_n_view = image_n_view.reshape(-1, image_n_view.shape[-3], image_n_view.shape[-2], image_n_view.shape[-1])
                image_n_view = image_n_view.float().to(device)

                canonical_images = canonical_images.float().to(device)
                batch_size = images.shape[0]

                recon_images, latent_features = model(images)
                recon_loss = mse_loss(recon_images, canonical_images)

                latent_loss = None

                if config.latent_loss == 'supcontrast' or config.latent_loss == 'SimCLR':
                    _, latent_features = model(image_n_view) # (batch_size * n_views, latent_dim)
                    latent_features = latent_features.contiguous().view(batch_size, config.n_views, -1) # (batch_size, n_views, latent_dim)

                    # Both `labels` and `mask` are None, perform SimCLR unsupervised loss:
                    latent_loss = supcontrast_loss(features=latent_features)

                elif config.latent_loss == 'triplet':
                    _, anchor_features = model(images) # (batch_size, C_dim, H_dim, W_dim)
                    _, other_features = model(image_n_view) # (batch_size * n_views, C_dim, H_dim, W_dim)

                    anchor_features = anchor_features.contiguous().view(batch_size, -1) # (batch_size, latent_dim)
                    pos_neg_features = other_features.contiguous().view(batch_size, config.num_pos + config.num_neg, -1)
                    pos_features = pos_neg_features[:, :config.num_pos, :] # (batch_size, num_pos, latent_dim)
                    neg_features = pos_neg_features[:, config.num_pos:, :] # (batch_size, num_neg, latent_dim)

                    latent_loss = triplet_loss(anchor=anchor_features, positive=pos_features, negative=neg_features)
                else:
                    raise ValueError('`config.latent_loss`: %s not supported.' % config.latent_loss)

                val_latent_loss += latent_loss.item()
                val_recon_loss += recon_loss.item()

        val_latent_loss /= (iter_idx + 1)
        val_recon_loss /= (iter_idx + 1)
        val_loss = val_latent_loss + val_recon_loss

        log('Validation [%s/%s] loss: %.3f, latent: %.3f, recon: %.3f'
            % (epoch_idx + 1, config.max_epochs, val_loss, val_latent_loss,
               val_recon_loss),
            filepath=config.log_path,
            to_console=True)
        # if wandb_run is not None:
        #     wandb.log({'val/loss': val_loss,
        #                'val/latent_loss': val_latent_loss,
        #                'val/recon_loss': val_recon_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(config.DiffeoInvariantNet_model_save_path)
            log('%s: Model weights successfully saved.' % config.DiffeoInvariantNet_model,
                filepath=config.log_path,
                to_console=True)

        if early_stopper.step(val_loss):
            log('Early stopping criterion met. Ending training.',
                filepath=config.log_path,
                to_console=True)
            break

    return

@torch.no_grad()
def test(config):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    dataset, train_set, val_set, test_set = prepare_dataset(config=config)

    # Build the model
    try:
        model = globals()[config.DiffeoInvariantNet_model](num_filters=config.num_filters,
                                                           in_channels=3,
                                                           out_channels=3)
    except:
        raise ValueError('`config.DiffeoInvariantNet_model`: %s not supported.' % config.DiffeoInvariantNet_model)
    model = model.to(device)

    os.makedirs(config.output_save_path, exist_ok=True)
    save_path_fig_embeddings_inst = '%s/embeddings_inst.png' % config.output_save_path
    save_path_fig_reconstructed = '%s/reconstructed.png' % config.output_save_path

    model.load_weights(config.DiffeoInvariantNet_model_save_path, device=device)
    log('%s: Model weights successfully loaded.' % config.DiffeoInvariantNet_model,
        to_console=True)

    if config.latent_loss in ['SimCLR']:
        contrastive_loss = SupConLoss(temperature=config.temp,
                                      base_temperature=config.base_temp,
                                      contrastive_mode=config.contrastive_mode)
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
        for iter_idx, (images, _, image_n_view, _, canonical_images, _) in enumerate(test_set):
            images = images.float().to(device)  # [batch_size, C, H, W]

            # [batch_size, n_views, C, H, W] -> [batch_size * n_views, C, H, W]
            image_n_view = image_n_view.reshape(-1, image_n_view.shape[-3], image_n_view.shape[-2], image_n_view.shape[-1])
            image_n_view = image_n_view.float().to(device)

            canonical_images = canonical_images.float().to(device)
            batch_size = images.shape[0]

            recon_images, latent_features = model(images)
            recon_loss = mse_loss(recon_images, canonical_images)

            latent_loss = None
            if config.latent_loss == 'SimCLR':
                _, latent_features = model(image_n_view) # (batch_size * n_views, latent_dim)
                latent_features = latent_features.contiguous().view(batch_size, config.n_views, -1) # (batch_size, n_views, latent_dim)

                # Both `labels` and `mask` are None, perform SimCLR unsupervised loss:
                latent_loss = contrastive_loss(features=latent_features)

            elif config.latent_loss == 'triplet':
                _, anchor_features = model(images) # (batch_size, C_dim, H_dim, W_dim)
                _, other_features = model(image_n_view) # (batch_size * n_views, C_dim, H_dim, W_dim)

                anchor_features = anchor_features.contiguous().view(batch_size, -1) # (batch_size, latent_dim)
                pos_neg_features = other_features.contiguous().view(batch_size, config.num_pos + config.num_neg, -1)
                pos_features = pos_neg_features[:, :config.num_pos, :] # (batch_size, num_pos, latent_dim)
                neg_features = pos_neg_features[:, config.num_pos:, :] # (batch_size, num_neg, latent_dim)

                latent_loss = triplet_loss(anchor=anchor_features, positive=pos_features, negative=neg_features)

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
        filepath=config.log_path,
        to_console=False)


    # Visualize latent embeddings.
    embeddings = {split: None for split in ['train', 'val', 'test']}
    embedding_patch_id_int = {split: [] for split in ['train', 'val', 'test']}
    orig_inputs = {split: None for split in ['train', 'val', 'test']}
    canonical = {split: None for split in ['train', 'val', 'test']}
    reconstructed = {split: None for split in ['train', 'val', 'test']}

    with torch.no_grad():
        for split, split_set in zip(['train', 'val', 'test'], [train_set, val_set, test_set]):
            for iter_idx, (images, _, image_n_view, _, canonical_images, _) in enumerate(split_set):
                images = images.float().to(device)  # [batch_size, C, H, W]

                recon_images, latent_features = model(images)
                latent_features = torch.flatten(latent_features, start_dim=1)

                # Move to cpu to save memory on gpu.
                images = images.cpu()
                recon_images = recon_images.cpu()
                latent_features = latent_features.cpu()

                if embeddings[split] is None:
                    embeddings[split] = latent_features  # (batch_size, latent_dim)
                else:
                    embeddings[split] = torch.cat([embeddings[split], latent_features], dim=0)
                if reconstructed[split] is None:
                    reconstructed[split] = recon_images # (batch_size, in_chan, H, W)
                else:
                    reconstructed[split] = torch.cat([reconstructed[split], recon_images], dim=0)
                if orig_inputs[split] is None:
                    orig_inputs[split] = images
                else:
                    orig_inputs[split] = torch.cat([orig_inputs[split], images], dim=0)
                if canonical[split] is None:
                    canonical[split] = canonical_images
                else:
                    canonical[split] = torch.cat([canonical[split], canonical_images], dim=0)
                embedding_patch_id_int[split].extend([dataset.get_patch_id_idx(img_path=img_path) for img_path in img_paths])


            embeddings[split] = embeddings[split].numpy()
            reconstructed[split] = reconstructed[split].numpy()
            orig_inputs[split] = orig_inputs[split].numpy()
            canonical[split] = canonical[split].numpy()
            embedding_patch_id_int[split] = np.array(embedding_patch_id_int[split])
            assert len(embeddings[split]) == len(reconstructed[split]) == len(embedding_patch_id_int[split])

    # Quantify latent embedding quality.
    ins_clustering_acc = {}
    ins_topk_acc, ins_mAP = {}, {}

    for split in ['train', 'val', 'test']:
        if config.latent_loss == 'triplet':
            distance_measure = 'cosine'
        elif config.latent_loss == 'SimCLR':
            distance_measure = 'cosine'

        instance_adj = np.zeros((len(embedding_patch_id_int[split]), len(embedding_patch_id_int[split])))
        log(f'Constructing instance adjacency ({instance_adj.shape}) matrices...', to_console=True)
        for i in range(len(embedding_patch_id_int[split])):
            for j in range(len(embedding_patch_id_int[split])):
                # same patch id means same instance
                if embedding_patch_id_int[split][i] == embedding_patch_id_int[split][j]:
                    instance_adj[i, j] = 1

        print('before yo=======!')
        log(f'Done constructing instance adjacency ({instance_adj.shape}) matrices.', to_console=True)
        print('after yo=======!')

        ins_clustering_acc[split] = clustering_accuracy(embeddings[split],
                                                        embeddings['train'],
                                                        embedding_patch_id_int[split],
                                                        embedding_patch_id_int['train'],
                                                        distance_measure=distance_measure,
                                                        voting_k=1)
        ins_topk_acc[split] = topk_accuracy(embeddings[split],
                                            instance_adj,
                                            distance_measure=distance_measure,
                                            k=3)

        ins_mAP[split] = embedding_mAP(embeddings[split],
                                       instance_adj,
                                       distance_op=distance_measure)
        print('?===========')
        print(ins_clustering_acc)
        log(f'[{split}]Instance clustering accuracy: {ins_clustering_acc[split]:.3f}', to_console=True)
        log(f'[{split}]Instance top-k accuracy: {ins_topk_acc[split]:.3f}', to_console=True)
        log(f'[{split}]Instance mAP: {ins_mAP[split]:.3f}', to_console=True)

    return

# def infer(config):
#     '''
#         Inference mode. Given test image patch folder, load the model and
#         pair the test images with closest images in anchor bank.
#         The anchor patch bank: training images from the original or augmented images.
#         !TODO: we may only want to use the original images for the anchor bank,
#         !TODO: since the augmented images are not real instances. and the DiffeoMappingNet
#         !TODO: model is trained on the warping aug to original images.
#         Output: a csv file with the following columns:
#             - test_image_path
#             - closest_image_path
#             - distance
#             - source (original or augmented)
#     '''
#     # Step 1: Load the model & generate embeddings for images in anchor bank.

#     # Load anchor bank data
#     device = torch.device(
#         'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
#     aug_lists = config.aug_methods.split(',')
#     if config.dataset_name == 'MoNuSeg':
#         dataset = AugmentedMoNuSegDataset(augmentation_methods=aug_lists,
#                                             base_path=config.dataset_path,
#                                             target_dim=config.target_dim)
#     elif config.dataset_name == 'GLySAC':
#         dataset = AugmentedGLySACDataset(augmentation_methods=aug_lists,
#                                          base_path=config.dataset_path,
#                                          target_dim=config.target_dim)
#     else:
#         raise ValueError('`dataset_name`: %s not supported.' % config.dataset_name)

#     dataloader = DataLoader(dataset=dataset,
#                             batch_size=config.batch_size,
#                             shuffle=False,
#                             num_workers=config.num_workers)

#     # Build the model
#     try:
#         model = globals()[config.DiffeoInvariantNet_model](num_filters=config.num_filters,
#                                                            in_channels=3,
#                                                            out_channels=3)
#     except:
#         raise ValueError('`config.DiffeoInvariantNet_model`: %s not supported.' % config.DiffeoInvariantNet_model)
#     model = model.to(device)
#     model.load_weights(config.DiffeoInvariantNet_model_save_path, device=device)
#     log('%s: Model weights successfully loaded.' % config.DiffeoInvariantNet_model,
#         to_console=True)

#     # Generate embeddings for anchor bank
#     log('Generating embeddings for anchor bank. Total images: %d' % len(dataset),
#         to_console=True)
#     anchor_bank = {
#         'embeddings': [],
#         'img_paths': [],
#         'sources': [],
#     }
#     log('Inferring ...', to_console=True)

#     model.eval()
#     config.anchor_only = False # !FIXME: add as param
#     with torch.no_grad():
#         for iter_idx, (images, _, image_n_view, _, canonical_images, _) in enumerate(dataloader):
#             images = images.float().to(device)
#             _, latent_features = model(images)
#             latent_features = torch.flatten(latent_features, start_dim=1)
#             #print('latent_features.shape: ', latent_features.shape) # (batch_size, latent_dim)

#             for i in range(len(latent_features)):
#                 if config.anchor_only:
#                     if 'original' in img_paths[i]:
#                         anchor_bank['embeddings'].append(latent_features[i].cpu().numpy())
#                         anchor_bank['img_paths'].append(img_paths[i])
#                         anchor_bank['sources'].append('original')
#                 else:
#                     anchor_bank['embeddings'].append(latent_features[i].cpu().numpy())
#                     anchor_bank['img_paths'].append(img_paths[i])
#                     anchor_bank['sources'].append('original' if 'original' in img_paths[i] else 'augmented')

#     anchor_bank['embeddings'] = np.concatenate([anchor_bank['embeddings']], axis=0) # (N, latent_dim)
#     print('anchor_bank[embeddings].shape: ', anchor_bank['embeddings'].shape)
#     print('len(anchor_bank[img_paths]): ', len(anchor_bank['img_paths']))
#     print('len(anchor_bank[sources]): ', len(anchor_bank['sources']))
#     assert anchor_bank['embeddings'].shape[0] == len(anchor_bank['img_paths']) == len(anchor_bank['sources'])
#     log(f'Anchor bank embeddings generated. shape:{anchor_bank["embeddings"].shape}', to_console=True)

#     # Step 2: Generate embeddings for test images.
#     test_img_folder = os.path.join(config.test_folder, 'image')
#     test_img_files = sorted(glob(os.path.join(test_img_folder, '*.png')))
#     # Filter out on organ type
#     if config.dataset_name == 'MoNuSeg':
#         from preprocessing.Metas import Organ2FileID
#         file_ids = Organ2FileID[config.organ]['test']
#     elif config.dataset_name == 'GLySAC':
#         from preprocessing.Metas import GLySAC_Organ2FileID
#         file_ids = GLySAC_Organ2FileID[config.organ]['test']

#     test_img_files = [x for x in test_img_files if any([f'{file_id}' in x for file_id in file_ids])]

#     print('test_img_folder: ', test_img_folder)
#     test_img_bank = {
#         'embeddings': [],
#     }
#     test_images = [torch.Tensor(load_image(img_path, config.target_dim)) for img_path in test_img_files]
#     test_images = torch.stack(test_images, dim=0) # (N, in_chan, H, W)
#     log(f'Done loading test images, shape: {test_images.shape}', to_console=True)

#     test_dataset = torch.utils.data.TensorDataset(test_images)
#     test_loader = DataLoader(dataset=test_dataset,
#                              batch_size=config.batch_size,
#                              shuffle=False, # No shuffle since we're using the indices
#                              num_workers=config.num_workers)

#     with torch.no_grad():
#         for iter_idx, (images,) in enumerate(test_loader):
#             images = images.float().to(device)
#             _, latent_features = model(images)
#             latent_features = torch.flatten(latent_features, start_dim=1)
#             test_img_bank['embeddings'].extend([latent_features.cpu().numpy()])

#     test_img_bank['embeddings'] = np.concatenate(test_img_bank['embeddings'], axis=0) # (N, latent_dim)
#     log(f"test_img_bank[embeddings].shape: {test_img_bank['embeddings'].shape}", to_console=True)
#     log('Done generating embeddings for test images.', to_console=True)

#     # Step 3: Pair the test images with closest images in anchor bank.
#     if config.latent_loss == 'triplet':
#         distance_measure = 'cosine'
#     elif config.latent_loss == 'SimCLR':
#         distance_measure = 'cosine'
#     else:
#         raise ValueError('`config.latent_loss`: %s not supported.' % config.latent_loss)

#     log('Computing pairwise distances...', to_console=True)
#     print('test_img_bank[embeddings].shape: ', test_img_bank['embeddings'].shape)
#     print('anchor_bank[embeddings].shape: ', anchor_bank['embeddings'].shape)
#     dist_matrix = sklearn.metrics.pairwise_distances(test_img_bank['embeddings'],
#                                         anchor_bank['embeddings'],
#                                         metric=distance_measure) # [N, M]
#     closest_anchor_idxs = list(np.argmin(dist_matrix, axis=1, keepdims=False)) # [N]
#     closest_img_paths = [anchor_bank['img_paths'][idx] for idx in closest_anchor_idxs] # [N]

#     # Step 4: Save the results to a csv file.
#     results_df = pd.DataFrame({
#         'test_image_path': test_img_files,
#         'closest_image_path': closest_img_paths,
#         'distance': np.min(dist_matrix, axis=1, keepdims=False),
#         'source': [anchor_bank['sources'][idx] for idx in closest_anchor_idxs],
#     })

#     # overwriting the previous results
#     if os.path.exists(os.path.join(config.output_save_path, 'infer_pairs.csv')):
#         os.remove(os.path.join(config.output_save_path, 'infer_pairs.csv'))
#     results_df.to_csv(os.path.join(config.output_save_path, 'infer_pairs.csv'), index=False)

#     print('Results saved to: ', os.path.join(config.output_save_path, 'infer_pairs.csv'))

#     return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--mode', help='train|test|infer?', default='train')
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)
    parser.add_argument('--num-workers', help='Number of workers, e.g. use number of cores', default=4, type=int)

    parser.add_argument('--target-dim', default='(32, 32)', type=ast.literal_eval)
    parser.add_argument('--random-seed', default=1, type=int)

    parser.add_argument('--dataset-path', default='$ROOT/data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00_axis_patch_96x96/', type=str)
    parser.add_argument('--model-save-folder', default='$ROOT/checkpoints/', type=str)
    parser.add_argument('--output-save-folder', default='$ROOT/results/', type=str)

    parser.add_argument('--DiffeoInvariantNet-model', default='AutoEncoder', type=str)
    parser.add_argument('--dataset-name', default='A28+axis', type=str)
    parser.add_argument('--percentage', default=100, type=float)
    parser.add_argument('--organ', default=None, type=str)

    parser.add_argument('--learning-rate', default=1e-3, type=float)
    parser.add_argument('--patience', default=50, type=int)
    parser.add_argument('--depth', default=4, type=int)
    parser.add_argument('--latent-loss', default='SimCLR', type=str)
    parser.add_argument('--margin', default=0.2, type=float, help='Only relevant if latent-loss is `triplet`.')
    parser.add_argument('--temp', default=1.0, type=float)
    parser.add_argument('--base-temp', default=1.0, type=float)
    parser.add_argument('--n-views', default=2, type=int)
    parser.add_argument('--num-pos', default=1, type=int)
    parser.add_argument('--num-neg', default=1, type=int)
    parser.add_argument('--contrastive-mode', default='one', type=str)
    parser.add_argument('--aug-methods', default='rotation,uniform_stretch,directional_stretch,volume_preserving_stretch,partial_stretch', type=str)
    parser.add_argument('--max-epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--num-filters', default=32, type=int)
    parser.add_argument('--train-val-test-ratio', default='6:2:2', type=str)

    config = parser.parse_args()
    assert config.mode in ['train', 'test', 'infer']

    # handle edge cases.
    if config.latent_loss == 'triplet':
        config.n_views = config.num_pos + config.num_neg
        print('Setting `n_views` as `num_pos` + `num_neg` since we are using Triplet loss.')

    # fix path issues
    ROOT = '/'.join(
        os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
    for key in vars(config).keys():
        if type(getattr(config, key)) == str and '$ROOT' in getattr(config, key):
            setattr(config, key, getattr(config, key).replace('$ROOT', ROOT))

    model_name = f'dataset-{config.dataset_name}_fewShot-{config.percentage:.1f}%_organ-{config.organ}'
    DiffeoInvariantNet_str = f'DiffeoInvariantNet_model-{config.DiffeoInvariantNet_model}_depth-{config.depth}_latentLoss-{config.latent_loss}_seed{config.random_seed}'
    config.DiffeoInvariantNet_model_save_path = os.path.join(config.model_save_folder, model_name, DiffeoInvariantNet_str + '.ckpt')
    config.output_save_path = os.path.join(config.output_save_folder, model_name, DiffeoInvariantNet_str, '')
    config.log_path = os.path.join(config.output_save_folder, model_name, DiffeoInvariantNet_str + 'log.txt')

    print(config)

    seed_everything(config.random_seed)

    if config.mode == 'train':
        # Initialize log file.
        log_str = 'Config: \n'
        for key in vars(config).keys():
            log_str += '%s: %s\n' % (key, getattr(config, key))
        log_str += '\nTraining History:'
        log(log_str, filepath=config.log_path, to_console=True)

        train(config=config)
        test(config=config)
    elif config.mode == 'test':
        test(config=config)
    elif config.mode == 'infer':
        infer(config=config)

    print('Done.\n')