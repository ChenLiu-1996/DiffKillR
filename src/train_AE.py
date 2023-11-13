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
from utils.metrics import psnr, ssim
from utils.parse import parse_settings
from utils.seed import seed_everything
from utils.early_stop import EarlyStopping
from loss.supervised_contrastive import SupConLoss


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

    supercontrast_loss = SupConLoss(temperature=config.temp, 
                                    base_temperature=config.base_temp,
                                    contrast_mode=config.contrast_mode)
    mse_loss = torch.nn.MSELoss()
    early_stopper = EarlyStopping(mode='min',
                                  patience=config.patience,
                                  percentage=False)

    best_val_loss = np.inf

    for epoch_idx in tqdm(range(config.max_epochs)):
        train_loss, train_contrastive_loss, train_recon_loss = 0, 0, 0
        model.train()
        for iter_idx, (images, _, canonical_images, _, img_paths) in enumerate(tqdm(train_set)):
            images = images.float().to(device) # (bsz, in_chan, H, W)
            canonical_images = canonical_images.float().to(device)

            '''
                Reconstruction loss.
            '''
            recon_images, latent_features = model(images)
            recon_loss = mse_loss(recon_images, canonical_images)


            '''
                Supervised contrastive loss.
            '''
            # Construct batch_images [bsz * n_views, in_chan, H, W].
            batch_images = None
            cell_type_labels = []
            n_views = config.n_views
            bsz = images.shape[0]
            for image, img_path in zip(images, img_paths):
                cell_type = dataset.get_celltype(img_path=img_path)
                cell_type_labels.append(dataset.cell_type_to_idx[cell_type])
                if n_views > 1:
                    aug_images, aug_labels = dataset.sample_celltype(split='train',
                                                                     celltype=cell_type, 
                                                                     cnt=n_views-1)
                    aug_images = torch.Tensor(aug_images).to(device)
                    aug_labels = torch.Tensor(aug_labels).to(device)

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
            
            batch_images = batch_images.float().to(device)

            _, latent_features = model(batch_images) # (bsz * n_views, latent_dim)
            latent_features = latent_features.contiguous().view(bsz, n_views, -1) # (bsz, n_views, latent_dim)
            cell_type_labels = torch.tensor(cell_type_labels).to(device) # (bsz)

            contrastive_loss = supercontrast_loss(features=latent_features, 
                                                  labels=cell_type_labels)
            
            loss = contrastive_loss + recon_loss
            train_loss += loss.item()
            train_contrastive_loss += contrastive_loss.item()
            train_recon_loss += recon_loss.item()
            # print('\rIter %d, loss: %.3f, contrastive: %.3f, recon: %.3f\n' % (
            #     iter, loss.item(), contrastive_loss.item(), recon_loss.item()
            # ))

            # Simulate `config.batch_size` by batched optimizer update.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = train_loss / (iter_idx + 1) # avg loss over minibatches
        train_contrastive_loss = train_contrastive_loss / (iter_idx + 1)
        train_recon_loss = train_recon_loss / (iter_idx + 1)

        lr_scheduler.step()

        log('Train [%s/%s] loss: %.3f, contrastive: %.3f, recon: %.3f'
            % (epoch_idx + 1, config.max_epochs, train_loss, train_contrastive_loss,
               train_recon_loss),
            filepath=config.log_dir,
            to_console=False)

        # Validation.
        model.eval()
        with torch.no_grad():
            val_loss, val_contrastive_loss, val_recon_loss = 0, 0, 0
            for iter_idx, (images, _, canonical_images, _, img_paths) in enumerate(tqdm(val_set)):
                # NOTE: batch size is len(val_set) here.
                # May need to change this if val_set is too large.
                images = images.float().to(device)
                canonical_images = canonical_images.float().to(device)

                recon_images, latent_features = model(images)
                recon_loss = mse_loss(recon_images, canonical_images)

                # Construct batch_images [bsz * n_views, in_chan, H, W].
                batch_images = None
                cell_type_labels = []
                n_views = config.n_views
                bsz = images.shape[0]
                for image, img_path in zip(images, img_paths):
                    cell_type = dataset.get_celltype(img_path=img_path)
                    cell_type_labels.append(dataset.cell_type_to_idx[cell_type])
                    if n_views > 1:
                        aug_images, aug_labels = dataset.sample_celltype(split='val', # NOTE: use val
                                                                        celltype=cell_type, 
                                                                        cnt=n_views-1)
                        aug_images = torch.Tensor(aug_images).to(device)
                        aug_labels = torch.Tensor(aug_labels).to(device)

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
                
                batch_images = batch_images.float().to(device)
                
                _, latent_features = model(batch_images) # (bsz * n_views, latent_dim)
                latent_features = latent_features.contiguous().view(bsz, n_views, -1) # (bsz, n_views, latent_dim)
                cell_type_labels = torch.tensor(cell_type_labels).to(device) # (bsz)

                contrastive_loss = supercontrast_loss(features=latent_features, 
                                                      labels=cell_type_labels)

                val_contrastive_loss += contrastive_loss.item()
                val_recon_loss += recon_loss.item()
                val_loss += val_contrastive_loss + val_recon_loss

        val_loss /= (iter_idx + 1)
        val_contrastive_loss /= (iter_idx + 1)
        val_recon_loss /= (iter_idx + 1)

        log('Validation [%s/%s] loss: %.3f, contrastive: %.3f, recon: %.3f\n'
            % (epoch_idx + 1, config.max_epochs, val_loss, val_contrastive_loss, 
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
    dataset, _, _, test_set = prepare_dataset(config=config)

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

    save_path_fig_summary = '%s/results/summary.png' % config.output_save_path
    os.makedirs(os.path.dirname(save_path_fig_summary), exist_ok=True)

    supercontrast_loss = SupConLoss(temperature=config.temp, 
                                    base_temperature=config.base_temp,
                                    contrast_mode=config.contrast_mode)
    mse_loss = torch.nn.MSELoss()

    # Test.
    model.eval()
    with torch.no_grad():
        test_loss, test_contrastive_loss, test_recon_loss = 0, 0, 0
        for iter_idx, (images, _, canonical_images, _, img_paths) in enumerate(tqdm(test_set)):
            images = images.float().to(device)
            canonical_images = canonical_images.float().to(device)

            recon_images, latent_features = model(images)
            recon_loss = mse_loss(recon_images, canonical_images)

            # Construct batch_images [bsz * n_views, in_chan, H, W].
            batch_images = None
            cell_type_labels = []
            n_views = config.n_views
            bsz = images.shape[0]
            for image, img_path in zip(images, img_paths):
                cell_type = dataset.get_celltype(img_path=img_path)
                cell_type_labels.append(dataset.cell_type_to_idx[cell_type])
                if n_views > 1:
                    aug_images, aug_labels = dataset.sample_celltype(split='test', # NOTE: use test
                                                                    celltype=cell_type, 
                                                                    cnt=n_views-1)
                    aug_images = torch.Tensor(aug_images).to(device)
                    aug_labels = torch.Tensor(aug_labels).to(device)

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
            
            batch_images = batch_images.float().to(device)
            
            _, latent_features = model(batch_images) # (bsz * n_views, latent_dim)
            latent_features = latent_features.contiguous().view(bsz, n_views, -1) # (bsz, n_views, latent_dim)
            cell_type_labels = torch.tensor(cell_type_labels).to(device) # (bsz)

            contrastive_loss = supercontrast_loss(features=latent_features, 
                                                    labels=cell_type_labels)

            test_contrastive_loss += contrastive_loss.item()
            test_recon_loss += recon_loss.item()
            test_loss += test_contrastive_loss + test_recon_loss

    test_loss /= (iter_idx + 1)
    test_contrastive_loss /= (iter_idx + 1)
    test_recon_loss /= (iter_idx + 1)

    log('Test loss: %.3f, contrastive: %.3f, recon: %.3f\n'
        % (test_loss, test_contrastive_loss, test_recon_loss),
        filepath=config.log_dir,
        to_console=False)
    

    # Visualize latent embeddings.

    # Visualize input vs. reconstructed


    #...
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--mode', help='`train` or `test`?', required=True)
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config.gpu_id = args.gpu_id
    config = parse_settings(config, log_settings=args.mode == 'train')

    assert args.mode in ['train', 'test']

    seed_everything(config.random_seed)

    if args.mode == 'train':
        train(config=config)
        test(config=config)
    elif args.mode == 'test':
        test(config=config)
