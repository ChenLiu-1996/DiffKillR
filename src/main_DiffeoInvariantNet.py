'''
Main file for DiffeoInvariantNet.
'''

import argparse
import numpy as np
import torch
import sklearn.metrics
import pandas as pd
import wandb

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


def train(config, wandb_run=None):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = 'mps'

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
        warmup_epochs=config.max_epochs//5,
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

    loss_fn_mse = torch.nn.MSELoss()
    early_stopper = EarlyStopping(mode='min',
                                  patience=config.patience)

    best_val_loss = np.inf

    for epoch_idx in range(config.max_epochs):
        train_loss, train_latent_loss, train_recon_loss = 0, 0, 0
        model.train()
        dataset.set_deterministic(False)

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
            recon_loss = loss_fn_mse(recon_images, canonical_images)

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
                # For triplet loss, `n_views` is set to be `num_pos` + `num_neg`.

                _, anchor_features = model(images) # (batch_size, C_dim, H_dim, W_dim)
                _, other_features = model(image_n_view) # (batch_size * n_views, C_dim, H_dim, W_dim)

                anchor_features = anchor_features.contiguous().view(batch_size, -1) # (batch_size, latent_dim)
                pos_neg_features = other_features.contiguous().view(batch_size, config.num_pos + config.num_neg, -1) # (batch_size, n_views, latent_dim)
                pos_features = pos_neg_features[:, :config.num_pos, :] # (batch_size, num_pos, latent_dim)
                neg_features = pos_neg_features[:, config.num_pos:, :] # (batch_size, num_neg, latent_dim)
                # Permute the neg features along batch to make them negative samples.
                permuted_idx = np.random.permutation(len(neg_features))
                neg_features = neg_features[permuted_idx, ...]

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
        if wandb_run is not None:
            wandb.log({'train/loss': train_loss,
                       'train/latent_loss': train_latent_loss,
                       'train/recon_loss': train_recon_loss})

        # Validation.
        model.eval()
        dataset.set_deterministic(True)
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
                recon_loss = loss_fn_mse(recon_images, canonical_images)

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
                    pos_neg_features = other_features.contiguous().view(batch_size, config.num_pos + config.num_neg, -1) # (batch_size, n_views, latent_dim)
                    pos_features = pos_neg_features[:, :config.num_pos, :] # (batch_size, num_pos, latent_dim)
                    neg_features = pos_neg_features[:, config.num_pos:, :] # (batch_size, num_neg, latent_dim)
                    # Permute the neg features along batch to make them negative samples.
                    permuted_idx = np.random.permutation(len(neg_features))
                    neg_features = neg_features[permuted_idx, ...]

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
        if wandb_run is not None:
            wandb.log({'val/loss': val_loss,
                       'val/latent_loss': val_latent_loss,
                       'val/recon_loss': val_recon_loss})

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
    if torch.backends.mps.is_available():
        device = 'mps'

    dataset, train_loader, val_loader, test_loader = prepare_dataset(config=config)

    print('Len(Dataset): ', len(dataset))

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

    loss_fn_mse = torch.nn.MSELoss()

    # Test.
    model.eval()
    dataset.set_deterministic(True)
    with torch.no_grad():
        test_loss, test_latent_loss, test_recon_loss = 0, 0, 0
        for iter_idx, (images, _, image_n_view, _, canonical_images, _) in enumerate(test_loader):
            images = images.float().to(device)  # [batch_size, C, H, W]

            # [batch_size, n_views, C, H, W] -> [batch_size * n_views, C, H, W]
            image_n_view = image_n_view.reshape(-1, image_n_view.shape[-3], image_n_view.shape[-2], image_n_view.shape[-1])
            image_n_view = image_n_view.float().to(device)

            canonical_images = canonical_images.float().to(device)
            batch_size = images.shape[0]

            recon_images, latent_features = model(images)
            recon_loss = loss_fn_mse(recon_images, canonical_images)

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
                pos_neg_features = other_features.contiguous().view(batch_size, config.num_pos + config.num_neg, -1) # (batch_size, n_views, latent_dim)
                pos_features = pos_neg_features[:, :config.num_pos, :] # (batch_size, num_pos, latent_dim)
                neg_features = pos_neg_features[:, config.num_pos:, :] # (batch_size, num_neg, latent_dim)
                # Permute the neg features along batch to make them negative samples.
                permuted_idx = np.random.permutation(len(neg_features))
                neg_features = neg_features[permuted_idx, ...]

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
    reference_embeddings = {split: None for split in ['train', 'val', 'test']}
    n_views_embeddings = {split: None for split in ['train', 'val', 'test']}
    orig_inputs = {split: None for split in ['train', 'val', 'test']}
    canonical = {split: None for split in ['train', 'val', 'test']}
    reconstructed = {split: None for split in ['train', 'val', 'test']}

    with torch.no_grad():
        dataset.set_deterministic(False)  # To allow for variation.

        # Ensure no shuffling for train loader.
        train_loader_no_shuffle = DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            num_workers=train_loader.num_workers,
            collate_fn=train_loader.collate_fn,
            pin_memory=train_loader.pin_memory,
            drop_last=train_loader.drop_last,
            timeout=train_loader.timeout,
            worker_init_fn=train_loader.worker_init_fn,
            multiprocessing_context=train_loader.multiprocessing_context,
            generator=train_loader.generator,
            prefetch_factor=train_loader.prefetch_factor,
            persistent_workers=train_loader.persistent_workers,
            shuffle=False)

        for split, split_set in zip(['train', 'val', 'test'], [train_loader_no_shuffle, val_loader, test_loader]):
            # NOTE (quite arbitrary): Repeat for 5 random augmentations.
            for _ in range(1):
                for iter_idx, (images, _, image_n_view, _, canonical_images, _) in enumerate(split_set):
                    # NOTE (quite arbitrary): Limit the computation.
                    batch_size = images.shape[0]
                    if iter_idx * batch_size > 400:
                        break

                    # Infer latent embeddings for aug view.
                    images = images.float().to(device)  # [batch_size, C, H, W]
                    recon_images, latent_features = model(images)
                    latent_features = torch.flatten(latent_features, start_dim=1)

                    # Infer latent embeddings for canonical view.
                    canonical_images = canonical_images.float().to(device)
                    _, latent_features_ref = model(canonical_images)
                    latent_features_ref = torch.flatten(latent_features_ref, start_dim=1)

                    # Infer latent embeddings for n_views: [batch_size, n_views, C, H, W] -> [batch_size * n_views, C, H, W]
                    image_n_view = image_n_view.reshape(-1, image_n_view.shape[-3], image_n_view.shape[-2], image_n_view.shape[-1])
                    image_n_view = image_n_view.float().to(device)
                    _, latent_features_n_views = model(image_n_view)
                    latent_features_n_views = torch.flatten(latent_features_n_views, start_dim=1)

                    # Move to cpu to save memory on gpu.
                    images = images.cpu()
                    recon_images = recon_images.cpu()
                    latent_features = latent_features.cpu()
                    latent_features_ref = latent_features_ref.cpu()
                    latent_features_n_views = latent_features_n_views.cpu()

                    if embeddings[split] is None:
                        embeddings[split] = latent_features  # (batch_size, latent_dim)
                    else:
                        embeddings[split] = torch.cat([embeddings[split], latent_features], dim=0)

                    if reference_embeddings[split] is None:
                        reference_embeddings[split] = latent_features_ref
                    else:
                        reference_embeddings[split] = torch.cat([reference_embeddings[split], latent_features_ref], dim=0)

                    if n_views_embeddings[split] is None:
                        n_views_embeddings[split] = latent_features_n_views
                    else:
                        n_views_embeddings[split] = torch.cat([n_views_embeddings[split], latent_features_n_views], dim=0)

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

            embeddings[split] = embeddings[split].numpy()
            reference_embeddings[split] = reference_embeddings[split].numpy()
            reconstructed[split] = reconstructed[split].numpy()
            orig_inputs[split] = orig_inputs[split].cpu().numpy()
            canonical[split] = canonical[split].cpu().numpy()

    # Quantify latent embedding quality.
    ins_clustering_acc = {}
    ins_topk_acc = {k: {} for k in [3]}
    ins_mAP = {}

    for split in ['train', 'val', 'test']:
        if config.latent_loss == 'triplet':
            distance_measure = 'cosine'
        elif config.latent_loss == 'SimCLR':
            distance_measure = 'cosine'

        ins_clustering_acc[split] = clustering_accuracy(embeddings[split],
                                                        reference_embeddings[split],
                                                        labels=np.arange(len(embeddings[split])),
                                                        reference_labels=np.arange(len(reference_embeddings[split])),
                                                        distance_measure=distance_measure,
                                                        voting_k=1)

        # repeat the labels for n_views times.
        n_views_labels = np.repeat(np.arange(len(embeddings[split])), config.n_views)
        concatenated_labels = np.concatenate([np.arange(len(embeddings[split])), n_views_labels], axis=0)
        concatenated_embeddings = np.concatenate([embeddings[split], n_views_embeddings[split]], axis=0)
        instance_adj = np.zeros((len(concatenated_embeddings), len(concatenated_embeddings)))
        for i in range(len(concatenated_embeddings)):
            for j in range(len(concatenated_embeddings)):
                if concatenated_labels[i] == concatenated_labels[j]:
                    instance_adj[i, j] = 1
        print('Done constructing instance adjacency matrix of shape: ', instance_adj.shape)
        for k in ins_topk_acc.keys():
            ins_topk_acc[k][split] = topk_accuracy(concatenated_embeddings,
                                                    instance_adj,
                                                    distance_measure=distance_measure,
                                                    k=k)
        ins_mAP[split] = embedding_mAP(concatenated_embeddings,
                                        instance_adj,
                                        distance_op=distance_measure)

        log(f'[{split}]Instance clustering accuracy: {ins_clustering_acc[split]:.3f}', to_console=True)
        for k in ins_topk_acc.keys():
            log(f'[{split}]Instance top-{k} accuracy: {ins_topk_acc[k][split]:.3f}', to_console=True)
        log(f'[{split}]Instance mAP: {ins_mAP[split]:.3f}', to_console=True)
    return

def main(config):
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

    model_name = f'dataset-{config.dataset_name}_fewShot-{config.percentage}%_organ-{config.organ}'
    DiffeoInvariantNet_str = f'DiffeoInvariantNet-{config.DiffeoInvariantNet_model}_depth-{config.depth}_latentLoss-{config.latent_loss}_epoch-{config.max_epochs}_seed-{config.random_seed}_backgroundRatio-{config.background_ratio:.1f}'
    config.output_save_path = os.path.join(config.output_save_folder, model_name, DiffeoInvariantNet_str, '')
    config.DiffeoInvariantNet_model_save_path = os.path.join(config.output_save_path, model_name, DiffeoInvariantNet_str, 'model.ckpt')
    config.log_path = os.path.join(config.output_save_folder, model_name, DiffeoInvariantNet_str, 'log.txt')

    print(config)

    seed_everything(config.random_seed)

    # run scr/preprocessing/prepare_MoNuSeg.py first to generate train data.
    print('Running scr/preprocessing/prepare_MoNuSeg.py to generate train data...')
    script_path = ROOT + '/src/preprocessing/prepare_MoNuSeg.py'
    os.system(f"python {script_path} \
                --patch_size {config.target_dim[0] * 2} \
                --aug_patch_size {config.target_dim[0]} \
                --organ {config.organ} \
                --background_ratio {config.background_ratio}")
    print('Training data generated.')

    wandb_run = None
    if config.use_wandb and config.mode == 'train':
        wandb_run = wandb.init(
            entity=config.wandb_username,    # NOTE: need to use your wandb user name.
            project="cellseg",               # NOTE: need to create project on your wandb website.
            name=model_name + '_' + DiffeoInvariantNet_str,
            config=config,
            reinit=True,
            settings=wandb.Settings(start_method="thread")
        )

    if config.mode == 'train':
        # Initialize log file.
        log_str = 'Config: \n'
        for key in vars(config).keys():
            log_str += '%s: %s\n' % (key, getattr(config, key))
        log_str += '\nTraining History:'
        log(log_str, filepath=config.log_path, to_console=True)

        train(config=config, wandb_run=wandb_run)
        test(config=config)
    elif config.mode == 'test':
        test(config=config)

    if wandb_run is not None:
        wandb_run.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--mode', help='train|test|infer?', default='train')
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)
    parser.add_argument('--num-workers', help='Number of workers, e.g. use number of cores', default=4, type=int)

    parser.add_argument('--target-dim', default='(32, 32)', type=ast.literal_eval)
    parser.add_argument('--random-seed', default=1, type=int)

    # parser.add_argument('--model-save-folder', default='$ROOT/checkpoints/', type=str)
    parser.add_argument('--output-save-folder', default='$ROOT/results/', type=str)

    parser.add_argument('--DiffeoInvariantNet-model', default='AutoEncoder', type=str)
    parser.add_argument('--dataset-name', default='MoNuSeg', type=str)
    parser.add_argument('--dataset-path', default='$ROOT/data/MoNuSeg2018TrainData_patch_96x96', type=str)
    parser.add_argument('--percentage', default=100, type=float) # NOTE: this is the percentage of the training data.
    parser.add_argument('--organ', default='Breast', type=str)
    parser.add_argument('--background-ratio', default=1.0, type=float) # How many background patches to generate for each cell patch.

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

    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-username', default='yale-cl2482', type=str)

    config = parser.parse_args()

    if config.organ is not None:
        config.dataset_path = f'{config.dataset_path}_{config.organ}'

    main(config)
