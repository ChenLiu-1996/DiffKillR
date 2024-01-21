from typing import Tuple
import argparse
import numpy as np
import torch
import yaml
import os
# from model.scheduler import LinearWarmupCosineAnnealingLR
from tqdm import tqdm
from matplotlib import pyplot as plt
from model.unet import UNet
from registration.spatial_transformer import SpatialTransformer as Warper
from utils.attribute_hashmap import AttributeHashmap
from utils.prepare_dataset import prepare_dataset
from utils.log_util import log
from utils.parse import parse_settings
from utils.seed import seed_everything
from utils.early_stop import EarlyStopping


def numpy_variables(*tensors: torch.Tensor) -> Tuple[np.array]:
    '''
    Some repetitive numpy casting of variables.
    '''
    return [_tensor.cpu().detach().numpy().transpose(1, 2, 0) for _tensor in tensors]


def plot_side_by_side(save_path, im_I, im_C, im_I2C_C2I, im_I2C, lb_I, lb_C, lb_C2I):
    fig_sbs = plt.figure(figsize=(20, 8))

    ax = fig_sbs.add_subplot(2, 4, 1)
    ax.imshow(np.clip((im_I + 1) / 2, 0, 1))
    ax.set_title('Image (I)')
    ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, 4, 2)
    ax.imshow(np.clip((im_C + 1) / 2, 0, 1))
    ax.set_title('Canonical Image (C)')
    ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, 4, 5)
    ax.imshow(np.clip((im_I2C_C2I + 1) / 2, 0, 1))
    ax.set_title('Cycled Image (I->C->I)')
    ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, 4, 6)
    ax.imshow(np.clip((im_I2C + 1) / 2, 0, 1))
    ax.set_title('Warped Image (I->C)')
    ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, 4, 3)
    ax.imshow(np.clip(lb_I, 0, 1), cmap='gray')
    ax.set_title('Label (I)')
    ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, 4, 4)
    ax.imshow(np.clip(lb_C, 0, 1), cmap='gray')
    ax.set_title('Canonical Label (C)')
    ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, 4, 7)
    ax.imshow(np.clip(lb_C2I, 0, 1), cmap='gray')
    ax.set_title('Projected Label (C->I)')
    ax.set_axis_off()

    fig_sbs.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig_sbs.savefig(save_path)
    plt.close(fig=fig_sbs)

    return


def train(config: AttributeHashmap):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    _, train_set, val_set, _ = \
        prepare_dataset(config=config)

    # Build the model
    try:
        warp_predictor = globals()[config.model](num_filters=config.num_filters,
                                                 in_channels=6,
                                                 out_channels=4)
    except:
        raise ValueError('`config.model`: %s not supported.' % config.model)

    warper = Warper(size=config.target_dim)

    warp_predictor = warp_predictor.to(device)
    warper = warper.to(device)

    optimizer = torch.optim.AdamW(warp_predictor.parameters(), lr=config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    mse_loss = torch.nn.MSELoss()
    early_stopper = EarlyStopping(mode='min',
                                  patience=config.patience,
                                  percentage=False)

    best_val_loss = np.inf

    for epoch_idx in tqdm(range(config.max_epochs)):
        train_loss, train_loss_forward, train_loss_cyclic = 0, 0, 0

        warp_predictor.train()
        for iter_idx, (images, labels, canonical_images, canonical_labels, img_paths) in enumerate(tqdm(train_set)):
            shall_plot = iter_idx % 20 == 0

            if len(labels.shape) == 3:
                labels = labels[:, None, ...]
            if len(canonical_labels.shape) == 3:
                canonical_labels = canonical_labels[:, None, ...]

            images = images.float().to(device) # (bsz, in_chan, H, W)
            canonical_images = canonical_images.float().to(device)
            labels = labels.float().to(device)
            canonical_labels = canonical_labels.float().to(device)

            # Predict the warping field.
            warp_predicted = warp_predictor(torch.cat([canonical_images, images], dim=1))
            warp_field_forward = warp_predicted[:, :2, ...]
            warp_field_reverse = warp_predicted[:, 2:, ...]

            # Apply the warping field.
            images_I2C = warper(images, flow=warp_field_forward)
            images_I2C_C2I = warper(images_I2C, flow=warp_field_reverse)
            labels_C2I = warper(canonical_labels, flow=warp_field_reverse)

            if shall_plot:
                save_path_fig_sbs = '%s/train/figure_log_epoch%s_sample%s.png' % (
                    config.save_folder, str(epoch_idx).zfill(5), str(iter_idx).zfill(5))
                plot_side_by_side(save_path_fig_sbs, *numpy_variables(
                    images[0], canonical_images[0],
                    images_I2C_C2I[0], images_I2C[0],
                    labels[0], canonical_labels[0],
                    labels_C2I[0]))

            loss_forward = mse_loss(canonical_images, images_I2C)
            loss_cyclic = mse_loss(images, images_I2C_C2I)

            loss = loss_forward + loss_cyclic
            train_loss += loss.item()
            train_loss_forward += loss_forward.item()
            train_loss_cyclic += loss_cyclic.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= (iter_idx + 1)
        train_loss_forward /= (iter_idx + 1)
        train_loss_cyclic /= (iter_idx + 1)

        lr_scheduler.step()

        log('Train [%s/%s] loss: %.3f, forward: %.3f, cyclic: %.3f'
            % (epoch_idx + 1, config.max_epochs, train_loss, train_loss_forward,
               train_loss_cyclic),
            filepath=config.log_dir,
            to_console=False)

        # Validation.
        warp_predictor.eval()
        with torch.no_grad():
            val_loss, val_loss_forward, val_loss_cyclic = 0, 0, 0
            for iter_idx, (images, labels, canonical_images, canonical_labels, img_paths) in enumerate(tqdm(val_set)):
                shall_plot = iter_idx % 20 == 0

                # NOTE: batch size is len(val_set) here.
                # May need to change this if val_set is too large.

                if len(labels.shape) == 3:
                    labels = labels[:, None, ...]
                if len(canonical_labels.shape) == 3:
                    canonical_labels = canonical_labels[:, None, ...]

                images = images.float().to(device) # (bsz, in_chan, H, W)
                canonical_images = canonical_images.float().to(device)
                labels = labels.float().to(device)
                canonical_labels = canonical_labels.float().to(device)

                # Predict the warping field.
                warp_predicted = warp_predictor(torch.cat([canonical_images, images], dim=1))
                warp_field_forward = warp_predicted[:, :2, ...]
                warp_field_reverse = warp_predicted[:, 2:, ...]

                # Apply the warping field.
                images_I2C = warper(images, flow=warp_field_forward)
                images_I2C_C2I = warper(images_I2C, flow=warp_field_reverse)
                labels_C2I = warper(canonical_labels, flow=warp_field_reverse)

                if shall_plot:
                    save_path_fig_sbs = '%s/val/figure_log_epoch%s_sample%s.png' % (
                        config.save_folder, str(epoch_idx).zfill(5), str(iter_idx).zfill(5))
                    plot_side_by_side(save_path_fig_sbs, *numpy_variables(
                        images[0], canonical_images[0],
                        images_I2C_C2I[0], images_I2C[0],
                        labels[0], canonical_labels[0],
                        labels_C2I[0]))

                loss_forward = mse_loss(canonical_images, images_I2C)
                loss_cyclic = mse_loss(images, images_I2C_C2I)
                loss = loss_forward + loss_cyclic

                val_loss += loss.item()
                val_loss_forward += loss_forward.item()
                val_loss_cyclic += loss_cyclic.item()

        val_loss /= (iter_idx + 1)
        val_loss_forward /= (iter_idx + 1)
        val_loss_cyclic /= (iter_idx + 1)

        log('Validation [%s/%s] loss: %.3f, forward: %.3f, cyclic: %.3f\n'
            % (epoch_idx + 1, config.max_epochs, val_loss, val_loss_forward,
               val_loss_cyclic),
            filepath=config.log_dir,
            to_console=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            warp_predictor.save_weights(config.model_save_path)
            log('%s: Model weights successfully saved.' % config.model,
                filepath=config.log_dir,
                to_console=False)

        if early_stopper.step(val_loss):
            log('Early stopping criterion met. Ending training.',
                filepath=config.log_dir,
                to_console=True)
            break

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--mode', help='`train` or `test`?', required=True)
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)
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
    config = parse_settings(config, log_settings=args.mode == 'train')

    assert args.mode in ['train', 'test']

    seed_everything(config.random_seed)

    if args.mode == 'train':
        train(config=config)
        # test(config=config)
    elif args.mode == 'test':
        # test(config=config)
        pass
