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


def train(config: AttributeHashmap):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    train_set, val_set, test_set = \
        prepare_dataset(config=config)

    # Build the model
    try:
        model = globals()[config.model](num_filters=config.num_filters,
                                        in_channels=3,
                                        out_channels=3)
    except:
        raise ValueError('`config.model`: %s not supported.' % config.model)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=10,
        warmup_start_lr=float(config.learning_rate) / 100,
        max_epochs=config.max_epochs,
        eta_min=0)

    loss_fn = torch.nn.MSELoss()
    best_val_loss = np.inf

    for epoch_idx in tqdm(range(config.max_epochs)):
        train_loss, train_recon_psnr, train_recon_ssim = 0, 0, 0
        model.train()
        for iter_idx, (images, labels) in enumerate(tqdm(train_set)):


            loss = loss_recon + loss_embed
            train_loss += loss.item()

            # Simulate `config.batch_size` by batched optimizer update.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_set.dataset)
        train_recon_psnr = train_recon_psnr / len(train_set.dataset)
        train_recon_ssim = train_recon_ssim / len(train_set.dataset)

        lr_scheduler.step()

        log('Train [%s/%s] loss: %.3f, PSNR (recon): %.3f, SSIM (recon): %.3f'
            % (epoch_idx + 1, config.max_epochs, train_loss, train_recon_psnr,
               train_recon_ssim),
            filepath=config.log_dir,
            to_console=False)

        val_recon_psnr, val_recon_ssim = 0, 0
        model.eval()
        with torch.no_grad():
            for (images, labels) in tqdm(val_set):

                loss = loss_recon + loss_embed
                train_loss += loss.item()

        val_recon_psnr = val_recon_psnr / len(val_set.dataset)
        val_recon_ssim = val_recon_ssim / len(val_set.dataset)

        log('Validation [%s/%s] PSNR (recon): %.3f, SSIM (recon): %.3f'
            % (epoch_idx + 1, config.max_epochs, val_recon_psnr,
               val_recon_ssim),
            filepath=config.log_dir,
            to_console=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(config.model_save_path)
            log('%s: Model weights successfully saved.' % config.model,
                filepath=config.log_dir,
                to_console=False)

    return


@torch.no_grad()
def test(config: AttributeHashmap):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    train_set, val_set, test_set = \
        prepare_dataset(config=config)

    # Build the model
    try:
        model = globals()[config.model](num_filters=config.num_filters,
                                        in_channels=3,
                                        out_channels=3)
    except:
        raise ValueError('`config.model`: %s not supported.' % config.model)

    model.to(device)
    model.load_weights(config.model_save_path, device=device)
    log('%s: Model weights successfully loaded.' % config.model,
        to_console=True)

    save_path_fig_summary = '%s/results/summary.png' % config.output_save_path
    os.makedirs(os.path.dirname(save_path_fig_summary), exist_ok=True)

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
