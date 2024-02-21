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
from utils.metrics import dice_coeff, IoU


def numpy_variables(*tensors: torch.Tensor) -> Tuple[np.array]:
    '''
    Some repetitive numpy casting of variables.
    '''
    return [_tensor.cpu().detach().numpy().transpose(1, 2, 0) for _tensor in tensors]


def plot_side_by_side(save_path, im_U, im_A, im_U2A_A2U, im_U2A, ma_U, ma_A, ma_A2U):
    plt.rcParams['font.family'] = 'serif'
    fig_sbs = plt.figure(figsize=(20, 8))

    ax = fig_sbs.add_subplot(2, 4, 1)
    ax.imshow(np.clip((im_U + 1) / 2, 0, 1))
    ax.set_title('Unannotated Image (U)')
    ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, 4, 2)
    ax.imshow(np.clip((im_A + 1) / 2, 0, 1))
    ax.set_title('Annotated Image (A)')
    ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, 4, 5)
    ax.imshow(np.clip((im_U2A_A2U + 1) / 2, 0, 1))
    ax.set_title('Cycled Image (U->A->U)')
    ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, 4, 6)
    ax.imshow(np.clip((im_U2A + 1) / 2, 0, 1))
    ax.set_title('Warped Image (U->A)')
    ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, 4, 3)
    ax.imshow(np.clip(ma_U, 0, 1), cmap='gray')
    ax.set_title('Unannotated Mask (U)')
    ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, 4, 4)
    ax.imshow(np.clip(ma_A, 0, 1), cmap='gray')
    ax.set_title('Annotated Mask (A)')
    ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, 4, 7)
    ax.imshow(np.clip(ma_A2U, 0, 1), cmap='gray')
    ax.set_title('Projected Mask (A->U)')
    ax.set_axis_off()

    fig_sbs.suptitle('Dice (Mask(U), Mask(A)) = %.3f, Dice (Mask(U), Mask(A->U)) = %.3f' % (
        dice_coeff(ma_U, ma_A), dice_coeff(ma_U, ma_A2U)), fontsize=15)
    fig_sbs.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig_sbs.savefig(save_path)
    plt.close(fig=fig_sbs)

    return

def load_match_pairs(matched_pair_path: str, mode: str):
    '''
    Load the matched pairs from csv. file and return 
        - unnanotated_images
        - unannotated_masks # if mode is train; unavailable for infer.
        - annotated_images
        - annotated_masks
    '''


    pass

# FIXME!: I think we can even use another cycle loss: AM -> UM -> AM
# FIXME!: Also, if the augmented mask is good, we can use it as a target for the forward cycle: UM -> AM. 
def train(config: OmegaConf, wandb_run=None):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    _, train_set, val_set, _ = prepare_dataset(config=config)

    # Build the model
    try:
        warp_predictor = globals()[config.model](num_filters=config.num_filters,
                                                 in_channels=6,
                                                 out_channels=4)
    except:
        raise ValueError('`config.model`: %s not supported.' % config.model)

    warp_predictor = warp_predictor.to(device)

    warper = Warper(size=config.target_dim)
    warper = warper.to(device)

    optimizer = torch.optim.AdamW(warp_predictor.parameters(), lr=config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    mse_loss = torch.nn.MSELoss()
    early_stopper = EarlyStopping(mode='min',
                                  patience=config.patience,
                                  percentage=False)

    best_val_loss = np.inf
    for epoch_idx in tqdm(range(config.max_epochs)):
        train_loss, train_loss_forward, train_loss_cyclic = 0, 0, 0
        train_dice_ref_list, train_dice_seg_list = [], []

        warp_predictor.train()
        plot_freq = int(len(train_set) // config.n_plot_per_epoch)
        for iter_idx, (unannotated_images, unannotated_masks, annotated_images, annotated_masks, _, _) in enumerate(tqdm(train_set)):
            shall_plot = iter_idx % plot_freq == plot_freq - 1

            if len(unannotated_masks.shape) == 3:
                unannotated_masks = unannotated_masks[:, None, ...]
            if len(annotated_masks.shape) == 3:
                annotated_masks = annotated_masks[:, None, ...]

            unannotated_images = unannotated_images.float().to(device) # (bsz, in_chan, H, W)
            annotated_images = annotated_images.float().to(device)
            unannotated_masks = unannotated_masks.float().to(device)
            annotated_masks = annotated_masks.float().to(device)
            # Only care about the binary mask.
            annotated_masks = (annotated_masks > 0.5).float()
            unannotated_masks = (unannotated_masks > 0.5).float()

            # Predict the warping field.
            warp_predicted = warp_predictor(torch.cat([annotated_images, unannotated_images], dim=1))
            warp_field_forward = warp_predicted[:, :2, ...]
            warp_field_reverse = warp_predicted[:, 2:, ...]

            # Apply the warping field.
            images_U2A = warper(unannotated_images, flow=warp_field_forward)
            images_U2A_A2U = warper(images_U2A, flow=warp_field_reverse)
            masks_A2U = warper(annotated_masks, flow=warp_field_reverse)
            #print(masks_A2U.shape, unannotated_masks.shape, annotated_masks.shape)
            #print('check a few masks_A2U:', masks_A2U[0, ...], unannotated_masks[0, ...], annotated_masks[0, ...])
            # Compute Dice Coeff.
            for i in range(len(masks_A2U)):
                train_dice_ref_list.append(
                    dice_coeff((annotated_masks[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0),
                               (unannotated_masks[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0)))
                train_dice_seg_list.append(
                    dice_coeff((masks_A2U[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0),
                               (unannotated_masks[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0)))

            if shall_plot:
                save_path_fig_sbs = '%s/train/figure_log_epoch%s_sample%s.png' % (
                    config.save_folder, str(epoch_idx).zfill(5), str(iter_idx).zfill(5))
                plot_side_by_side(save_path_fig_sbs, *numpy_variables(
                    unannotated_images[0], annotated_images[0],
                    images_U2A_A2U[0], images_U2A[0],
                    unannotated_masks[0] > 0.5, annotated_masks[0] > 0.5,
                    masks_A2U[0] > 0.5))

            loss_forward = mse_loss(annotated_images, images_U2A)
            loss_cyclic = mse_loss(unannotated_images, images_U2A_A2U)

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

        log('Train [%s/%s] loss: %.3f, forward: %.3f, cyclic: %.3f, Dice coeff (ref): %.3f \u00B1 %.3f, Dice coeff (seg): %.3f \u00B1 %.3f.'
            % (epoch_idx + 1, config.max_epochs, train_loss, train_loss_forward, train_loss_cyclic,
               np.mean(train_dice_ref_list), np.std(train_dice_ref_list),
               np.mean(train_dice_seg_list), np.std(train_dice_seg_list)),
            filepath=config.log_dir,
            to_console=False)
        if wandb_run is not None:
            wandb_run.log({'train/loss': train_loss,
                           'train/loss_forward': train_loss_forward,
                           'train/loss_cyclic': train_loss_cyclic,
                           'train/dice_ref_mean': np.mean(train_dice_ref_list),
                           'train/dice_ref_std': np.std(train_dice_ref_list),
                           'train/dice_seg_mean': np.mean(train_dice_seg_list),
                           'train/dice_seg_std': np.std(train_dice_seg_list)})

        # Validation.
        warp_predictor.eval()
        with torch.no_grad():
            val_loss, val_loss_forward, val_loss_cyclic = 0, 0, 0
            val_dice_ref_list, val_dice_seg_list = [], []
            plot_freq = int(len(val_set) // config.n_plot_per_epoch)
            for iter_idx, (unannotated_images, unannotated_masks, annotated_images, annotated_masks, _, _) in enumerate(tqdm(val_set)):
                shall_plot = iter_idx % plot_freq == plot_freq - 1

                # NOTE: batch size is len(val_set) here.
                # May need to change this if val_set is too large.

                if len(unannotated_masks.shape) == 3:
                    unannotated_masks = unannotated_masks[:, None, ...]
                if len(annotated_masks.shape) == 3:
                    annotated_masks = annotated_masks[:, None, ...]

                unannotated_images = unannotated_images.float().to(device) # (bsz, in_chan, H, W)
                annotated_images = annotated_images.float().to(device)
                unannotated_masks = unannotated_masks.float().to(device)
                annotated_masks = annotated_masks.float().to(device)
                # Only care about the binary mask.
                annotated_masks = (annotated_masks > 0.5).float()
                unannotated_masks = (unannotated_masks > 0.5).float()

                # Predict the warping field.
                warp_predicted = warp_predictor(torch.cat([annotated_images, unannotated_images], dim=1))
                warp_field_forward = warp_predicted[:, :2, ...]
                warp_field_reverse = warp_predicted[:, 2:, ...]

                # Apply the warping field.
                images_U2A = warper(unannotated_images, flow=warp_field_forward)
                images_U2A_A2U = warper(images_U2A, flow=warp_field_reverse)
                masks_A2U = warper(annotated_masks, flow=warp_field_reverse)

                # Compute Dice Coeff.
                for i in range(len(masks_A2U)):
                    val_dice_ref_list.append(
                        dice_coeff((annotated_masks[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0),
                                   (unannotated_masks[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0)))
                    val_dice_seg_list.append(
                        dice_coeff((masks_A2U[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0),
                                   (unannotated_masks[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0)))

                if shall_plot:
                    save_path_fig_sbs = '%s/val/figure_log_epoch%s_sample%s.png' % (
                        config.save_folder, str(epoch_idx).zfill(5), str(iter_idx).zfill(5))
                    plot_side_by_side(save_path_fig_sbs, *numpy_variables(
                        unannotated_images[0], annotated_images[0],
                        images_U2A_A2U[0], images_U2A[0],
                        unannotated_masks[0] > 0.5, annotated_masks[0] > 0.5,
                        masks_A2U[0] > 0.5))

                loss_forward = mse_loss(annotated_images, images_U2A)
                loss_cyclic = mse_loss(unannotated_images, images_U2A_A2U)
                loss = loss_forward + loss_cyclic

                val_loss += loss.item()
                val_loss_forward += loss_forward.item()
                val_loss_cyclic += loss_cyclic.item()

        val_loss /= (iter_idx + 1)
        val_loss_forward /= (iter_idx + 1)
        val_loss_cyclic /= (iter_idx + 1)

        log('Validation [%s/%s] loss: %.3f, forward: %.3f, cyclic: %.3f, Dice coeff (ref): %.3f \u00B1 %.3f, Dice coeff (seg): %.3f \u00B1 %.3f.'
            % (epoch_idx + 1, config.max_epochs, val_loss, val_loss_forward, val_loss_cyclic,
               np.mean(val_dice_ref_list), np.std(val_dice_ref_list),
               np.mean(val_dice_seg_list), np.std(val_dice_seg_list)),
            filepath=config.log_dir,
            to_console=False)
        if wandb_run is not None:
            wandb_run.log({'val/loss': val_loss,
                           'val/loss_forward': val_loss_forward,
                           'val/loss_cyclic': val_loss_cyclic,
                           'val/dice_ref_mean': np.mean(val_dice_ref_list),
                           'val/dice_ref_std': np.std(val_dice_ref_list),
                           'val/dice_seg_mean': np.mean(val_dice_seg_list),
                           'val/dice_seg_std': np.std(val_dice_seg_list)})

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
    if wandb_run is not None:
        wandb_run.finish()

    return


@torch.no_grad()
def test(config: AttributeHashmap, n_plot_per_epoch: int = None):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    _, _, _, test_set = prepare_dataset(config=config)

    # Build the model
    try:
        warp_predictor = globals()[config.model](num_filters=config.num_filters,
                                                 in_channels=6,
                                                 out_channels=4)
    except:
        raise ValueError('`config.model`: %s not supported.' % config.model)

    warp_predictor.load_weights(config.model_save_path, device=device)
    warp_predictor = warp_predictor.to(device)

    warper = Warper(size=config.target_dim)
    warper = warper.to(device)

    mse_loss = torch.nn.MSELoss()

    test_loss, test_loss_forward, test_loss_cyclic = 0, 0, 0
    test_dice_ref_list, test_dice_seg_list = [], []

    warp_predictor.eval()
    if n_plot_per_epoch is not None:
        plot_freq = int(len(test_set) // n_plot_per_epoch)
    else:
        plot_freq = 1

    for iter_idx, (unannotated_images, unannotated_masks, annotated_images, annotated_masks, _, _) in enumerate(tqdm(test_set)):
        shall_plot = iter_idx % plot_freq == plot_freq - 1

        if len(unannotated_masks.shape) == 3:
            unannotated_masks = unannotated_masks[:, None, ...]
        if len(annotated_masks.shape) == 3:
            annotated_masks = annotated_masks[:, None, ...]

        unannotated_images = unannotated_images.float().to(device) # (bsz, in_chan, H, W)
        annotated_images = annotated_images.float().to(device)
        unannotated_masks = unannotated_masks.float().to(device)
        annotated_masks = annotated_masks.float().to(device)
        # Only care about the binary mask.
        annotated_masks = (annotated_masks > 0.5).float()
        unannotated_masks = (unannotated_masks > 0.5).float()

        # Predict the warping field.
        warp_predicted = warp_predictor(torch.cat([annotated_images, unannotated_images], dim=1))
        warp_field_forward = warp_predicted[:, :2, ...]
        warp_field_reverse = warp_predicted[:, 2:, ...]

        # Apply the warping field.
        images_U2A = warper(unannotated_images, flow=warp_field_forward)
        images_U2A_A2U = warper(images_U2A, flow=warp_field_reverse)
        masks_A2U = warper(annotated_masks, flow=warp_field_reverse)

        # Compute Dice Coeff.
        for i in range(len(masks_A2U)):
            test_dice_ref_list.append(
                dice_coeff((annotated_masks[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0),
                           (unannotated_masks[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0)))
            test_dice_seg_list.append(
                dice_coeff((masks_A2U[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0),
                           (unannotated_masks[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0)))

        if shall_plot:
            save_path_fig_sbs = '%s/test/figure_log_sample%s.png' % (
                config.save_folder, str(iter_idx).zfill(5))
            plot_side_by_side(save_path_fig_sbs, *numpy_variables(
                unannotated_images[0], annotated_images[0],
                images_U2A_A2U[0], images_U2A[0],
                unannotated_masks[0] > 0.5, annotated_masks[0] > 0.5,
                masks_A2U[0] > 0.5))

        loss_forward = mse_loss(annotated_images, images_U2A)
        loss_cyclic = mse_loss(unannotated_images, images_U2A_A2U)

        loss = loss_forward + loss_cyclic
        test_loss += loss.item()
        test_loss_forward += loss_forward.item()
        test_loss_cyclic += loss_cyclic.item()

    test_loss /= (iter_idx + 1)
    test_loss_forward /= (iter_idx + 1)
    test_loss_cyclic /= (iter_idx + 1)

    log('Test loss: %.3f, forward: %.3f, cyclic: %.3f, Dice coeff (ref): %.3f \u00B1 %.3f, Dice coeff (seg): %.3f \u00B1 %.3f.'
        % (test_loss, test_loss_forward, test_loss_cyclic,
           np.mean(test_dice_ref_list), np.std(test_dice_ref_list),
           np.mean(test_dice_seg_list), np.std(test_dice_seg_list)),
        filepath=config.log_dir,
        to_console=False)

    return

@torch.no_grad()
def infer(config):
    '''
        Given input pair of images, infer the warping field.
        pair input is a cvs file with columns:
            - test_image_path
            - closest_image_path
            - distance
            - source (original or augmented)
        e.g.:
        warp_predictor = warp_predictor(torch.cat([closest_image, test_image], dim=1))
        warp_field_forward = warp_predicted[:, :2, ...]
        warp_field_reverse = warp_predicted[:, 2:, ...]

        test_mask = warper(closest_mask, flow=warp_field_forward)
    '''
    # NOTE: maybe we can even train on fly, for each pair.
    import pandas as pd
    from datasets.augmented_MoNuSeg import load_image, load_mask

    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    _, _, _, test_set = prepare_dataset(config=config)

    # Build the model
    try:
        warp_predictor = globals()[config.model](num_filters=config.num_filters,
                                                 in_channels=6,
                                                 out_channels=4)
    except:
        raise ValueError('`config.model`: %s not supported.' % config.model)

    warp_predictor.load_weights(config.model_save_path, device=device)
    warp_predictor = warp_predictor.to(device)

    warper = Warper(size=config.target_dim)
    warper = warper.to(device)

    # Step 1: Load matched pairs.
    matched_df = pd.read_csv(config.matched_pair_path)
    test_image_paths = matched_df['test_image_path'].tolist()
    closest_image_paths = matched_df['closest_image_path'].tolist()
    closest_label_paths = [x.replace('image', 'label') for x in closest_image_paths]
    test_label_paths = [x.replace('image', 'label') for x in test_image_paths]


    # load the images.
    test_images = [torch.Tensor(load_image(p, config.target_dim)) for p in test_image_paths]
    closest_images = [torch.Tensor(load_image(p, config.target_dim)) for p in closest_image_paths]
    closest_masks = [torch.Tensor(load_mask(p, config.target_dim)) for p in closest_label_paths]
    test_masks = [np.expand_dims(load_mask(p, config.target_dim), 0) for p in test_label_paths]
    test_images = torch.stack(test_images, dim=0).to(device) # (N, in_chan, H, W)
    closest_images = torch.stack(closest_images, dim=0).to(device) # (N, in_chan, H, W)
    closest_masks = torch.stack(closest_masks, dim=0).to(device) # (N, H, W)
    # test_masks = torch.stack(test_masks, dim=0).numpy() # (N, H, W)

    dataset = torch.utils.data.TensorDataset(closest_images, test_images, closest_masks)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Step 2: Predict & Apply the warping field.
    warp_predictor.eval()
    pred_mask_list = []
    print(f'Starting inference for {test_images.shape[0]} test images ...')
    for iter_idx, (bclosest_images, btest_images, bclosest_masks) in enumerate(tqdm(dataloader)):
        # [N, H, W] -> [N, 1, H, W]
        if len(bclosest_masks.shape) == 3:
            bclosest_masks = bclosest_masks[:, None, ...]

        btest_images = btest_images.float().to(device)
        bclosest_images = bclosest_images.float().to(device)
        bclosest_masks = bclosest_masks.float().to(device)

        # Predict the warping field.
        warp_predicted = warp_predictor(torch.cat([bclosest_images, btest_images], dim=1))
        warp_field_forward = warp_predicted[:, :2, ...]
        warp_field_reverse = warp_predicted[:, 2:, ...]
        #print(warp_field_forward.shape, warp_field_reverse.shape)

        # Apply the warping field.
        #images_U2A = warper(btest_images, flow=warp_field_forward)
        pred_masks = warper(bclosest_masks, flow=warp_field_reverse)
        bpred_mask_list = [m.cpu().detach().numpy() for m in pred_masks]
        pred_mask_list.extend(bpred_mask_list)

    assert len(pred_mask_list) == len(test_masks)
    print('Completed inference.')
    print('len(test_masks), len(pred_mask_list): ', len(test_masks), len(pred_mask_list))
    
    # Step 3: Evaluation. Compute Dice Coeff.
    print(f'Computing Dice Coeff for {len(pred_mask_list)} total masks...')
    dice_list = []
    iou_list = []
    for i in range(len(pred_mask_list)):
        # print('test_masks[i].shape, pred_mask_list[i].shape: ', \
        #       test_masks[i].shape, pred_mask_list[i].shape)
        dice_list.append(
            dice_coeff((test_masks[i] > 0.5).transpose(1, 2, 0),
                       (pred_mask_list[i] > 0.5).transpose(1, 2, 0)))
        iou_list.append(
            IoU((test_masks[i] > 0.5).transpose(1, 2, 0),
                       (pred_mask_list[i] > 0.5).transpose(1, 2, 0)))
    
    log('[Eval] Dice coeff (seg): %.3f \u00B1 %.3f.'
        % (np.mean(dice_list), np.std(dice_list)),
        filepath=config.log_dir,
        to_console=True)
    log('[Eval] IoU (seg): %.3f \u00B1 %.3f.'
        % (np.mean(iou_list), np.std(iou_list)),
        filepath=config.log_dir,
        to_console=True)

    return

from omegaconf import OmegaConf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--mode', help='`train` or `test` or `infer` ?', default='train')
    parser.add_argument('--mode_config', help='Path to model config file', default='./config/MoNuSeg_reg2seg.yaml')
    parser.add_argument('--aiae_config', help='Path to model config file', default='./config/MoNuSeg_AIAE.yaml')
    parser.add_argument('--data_config', help='Path to data config file', default='./config/MoNuSeg_data.yaml')
    # parser.add_argument('--run_count', help='Provide this during testing!', default=None, type=int)
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)
    parser.add_argument('--num-workers', help='Number of workers, e.g. use number of cores', default=4, type=int)
    parser.add_argument('--use-wandb', help='Use wandb for logging', default=True, type=bool)
    args = parser.parse_args()

    model_config = OmegaConf.load(args.model_config)
    aiae_config = OmegaConf.load(args.aiae_config)
    data_config = OmegaConf.load(args.data_config)
    config = OmegaConf.merge(model_config, data_config, aiae_config)
    
    config.gpu_id = args.gpu_id
    config.num_workers = args.num_workers
    config.use_wandb = args.use_wandb

    print(config)
    seed_everything(config.random_seed)

    # args = AttributeHashmap(args)
    # config = AttributeHashmap(yaml.safe_load(open(args.config)))
    # config.config_file_name = args.config
    # config.gpu_id = args.gpu_id
    # config.num_workers = args.num_workers
    # config = parse_settings(config, log_settings=args.mode == 'train', run_count=args.run_count)
    assert args.mode in ['train', 'test', 'infer']

    seed_everything(config.random_seed)

    wandb_run = None
    import wandb
    if args.use_wandb and args.mode == 'train':
        wandb_run = wandb.init(
            entity=config.wandb_entity,
            project="cellseg",
            name="monuseg-reg2seg",
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
            reinit=True,
            settings=wandb.Settings(start_method="thread")
        )

    if args.mode == 'train':
        train(config=config, wandb_run=wandb_run)
        test(config=config)
    elif args.mode == 'test':
        test(config=config)
    elif args.mode == 'infer':
        infer(config=config)
