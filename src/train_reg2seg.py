from typing import Tuple, List
import argparse
import numpy as np
import torch
import cv2
import yaml
import os
# from model.scheduler import LinearWarmupCosineAnnealingLR
from tqdm import tqdm
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

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
    results = []
    
    # convert all tensors to have the same shape. [C, H, W]
    for i in range(len(tensors)):
        curr = tensors[i].cpu().detach().numpy()
        if len(curr.shape) == 2:
            curr = np.expand_dims(curr, 0)
        curr = curr.transpose(1, 2, 0)
        results.append(curr)

    # [N, C, H, W] -> [N, H, W, C] for visualization.
    assert all([len(t.shape) == 3 for t in results])
    return results

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

    fig_sbs.suptitle('IoU (Mask(U), Mask(A)) = %.3f, IoU (Mask(U), Mask(A->U)) = %.3f' % (
        IoU(ma_U, ma_A), IoU(ma_U, ma_A2U)), fontsize=15)
    fig_sbs.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig_sbs.savefig(save_path)
    plt.close(fig=fig_sbs)

    return

def load_match_pairs(matched_pair_path_root: str, mode: str, config):
    '''
    Load the matched pairs from csv. file and return 
        - unanotated_images
        - unannotated_masks
        - annotated_images
        - annotated_masks
        - unanonated_image_paths
    '''
    import pandas as pd
    from datasets.augmented_MoNuSeg import load_image, load_mask

    matched_df = pd.read_csv(os.path.join(matched_pair_path_root, f'{mode}_pairs.csv'))
    unanotated_image_paths = matched_df['test_image_path'].tolist()
    annotated_image_paths = matched_df['closest_image_path'].tolist()
    annotated_label_paths = [x.replace('image', 'label') for x in annotated_image_paths]
    unannotated_label_paths = [x.replace('image', 'label') for x in unanotated_image_paths]
    
    unannotated_images = [torch.Tensor(load_image(p, config.target_dim)) for p in unanotated_image_paths]
    annotated_images = [torch.Tensor(load_image(p, config.target_dim)) for p in annotated_image_paths]
    annotated_masks = [torch.Tensor(load_mask(p, config.target_dim)) for p in annotated_label_paths]
    unannotated_masks = [torch.Tensor(load_mask(p, config.target_dim)) for p in unannotated_label_paths]

    unannotated_images = torch.stack(unannotated_images, dim=0) # (N, in_chan, H, W)
    annotated_images = torch.stack(annotated_images, dim=0) # (N, in_chan, H, W)
    annotated_masks = torch.stack(annotated_masks, dim=0) # (N, H, W)
    unannotated_masks = torch.stack(unannotated_masks, dim=0)
    
    assert len(unannotated_images) == len(annotated_images) == len(annotated_masks)
    print(f'Loaded {len(unannotated_images)} pairs of images and masks.')
    
    return unannotated_images, unannotated_masks, \
        annotated_images, annotated_masks, unanotated_image_paths

# FIXME!: I think we can even use another cycle loss: AM -> UM -> AM
# FIXME!: Also, if the augmented mask is good, we can use it as a target for the forward cycle: UM -> AM. 
def train(config: OmegaConf, wandb_run=None):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    
    # Set all the paths.
    model_name = f'{config.percentage:.3f}_{config.organ}_m{config.multiplier}_MoNuSeg_depth{config.depth}_seed{config.random_seed}_{config.latent_loss}'
    matched_pair_path_root = os.path.join(config.output_save_root, model_name)
    config.log_dir = os.path.join(config.log_folder, model_name) # This is log file path.
    config.model_save_path = os.path.join(config.output_save_root, model_name, 'reg2seg.ckpt')
    
    # Load train, val, test sets
    load_results_map = {
        'train': load_match_pairs(matched_pair_path_root, mode='train', config=config),
        'val': load_match_pairs(matched_pair_path_root, mode='val', config=config),
        'test': load_match_pairs(matched_pair_path_root, mode='test', config=config)
    }
    dataset_map = {}
    dataloader_map = {}
    for k, load_results in load_results_map.items():
        (unannotated_images, unannotated_masks, annotated_images, annotated_masks, unannotated_images_paths) = load_results
        dataset = torch.utils.data.TensorDataset(unannotated_images, unannotated_masks, annotated_images, annotated_masks)
        loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        dataset_map[k] = dataset
        dataloader_map[k] = loader

    train_loader = dataloader_map['train']
    val_loader = dataloader_map['val']
    test_loader = dataloader_map['test']
    # merge val, test loader?
    val_test_set = torch.utils.data.ConcatDataset([dataset_map['val'], dataset_map['test']])
    val_test_loader = torch.utils.data.DataLoader(val_test_set, batch_size=config.batch_size, shuffle=True)

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
        plot_freq = int(len(dataset_map['train']) // config.n_plot_per_epoch)
        for iter_idx, (unannotated_images, unannotated_masks, annotated_images, annotated_masks) in enumerate(tqdm(train_loader)):
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
                save_folder = os.path.join(config.output_save_root, model_name, 'reg2seg')
                save_path_fig_sbs = '%s/train/figure_log_epoch%s_sample%s.png' % (
                    save_folder, str(epoch_idx).zfill(5), str(iter_idx).zfill(5))
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
            plot_freq = int(len(val_test_set) // config.n_plot_per_epoch)
            print('yo: ', plot_freq, len(val_test_loader))
            for iter_idx, (unannotated_images, unannotated_masks, annotated_images, annotated_masks) in enumerate(tqdm(val_test_loader)):
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
                    save_folder = os.path.join(config.output_save_root, model_name, 'reg2seg')
                    save_path_fig_sbs = '%s/val/figure_log_epoch%s_sample%s.png' % (
                        save_folder, str(epoch_idx).zfill(5), str(iter_idx).zfill(5))
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


import re
from glob import glob

def extract_h_w(file_path):
    h_w = re.findall('H(-?\d+)_W(-?\d+)', file_path)
    assert len(h_w) == 1

    h = int(h_w[0][0])
    w = int(h_w[0][1])
    
    return h, w

def stitch_patches(pred_mask_folder, stitched_size=(1000,1000)) -> Tuple[List[np.array], str]:
    '''
    Stitch the patches together.

    '''
    imsize = 32 # TODO: add as param
    stitched_folder = pred_mask_folder.replace('pred_patches', 'stitched_masks')
    colored_stitched_folder = pred_mask_folder.replace('pred_patches', 'colored_stitched_masks')
    os.makedirs(stitched_folder, exist_ok=True)
    os.makedirs(colored_stitched_folder, exist_ok=True)

    print('pred_mask_folder: ', pred_mask_folder, ' stitched_folder: ', stitched_folder)

    mask_list = sorted(glob(pred_mask_folder + '/*.png'))

    base_mask_list = []
    for mask_path in mask_list:
        # print('mask_path: ', mask_path)
        h, w = extract_h_w(mask_path)
        h_w_string = f'_H{h}_W{w}'
        # print('mask_path: ', mask_path, h, w)
        h_w_string_idx = mask_path.find(h_w_string)
        base_mask_path = mask_path[:h_w_string_idx] + '.png'

        if base_mask_path not in base_mask_list:
            # print('[True] base_mask_path: ', base_mask_path)
            base_mask_list.append(base_mask_path)
    

    stitched_mask_list = []

    for base_mask_path in base_mask_list:
        # print('[Stitching] base_mask_path: ', base_mask_path)        
        mask_stitched = np.zeros((stitched_size[0], stitched_size[1]), dtype=np.uint8)
        mask_patch_list = [item for item in mask_list if base_mask_path.replace('.png', '') in item]
        
        for mask_patch_path in mask_patch_list:
            h, w = extract_h_w(mask_patch_path)

            offset_h = min(0, h) # negative in case of negative h or w
            offset_w = min(0, w)
            start_h = max(0, h)
            start_w = max(0, w)
            end_h = min(start_h + imsize + offset_h, stitched_size[0])
            end_w = min(start_w + imsize + offset_w, stitched_size[1])
            actual_height = end_h - start_h
            actual_width = end_w - start_w

            # print('mask_patch h, w: ', h, w)
            # print('start_h, end_h, start_w, end_w: ', start_h, end_h, start_w, end_w)
            mask_patch = cv2.imread(mask_patch_path, cv2.IMREAD_GRAYSCALE)
            new_patch = mask_patch[-offset_h:-offset_h + actual_height, -offset_w:-offset_w + actual_width]
            old_patch = mask_stitched[start_h:end_h, start_w:end_w]
            # print('old_patch.shape: ', old_patch.shape, ' mask_patch.shape: ', mask_patch.shape, \
            #       ' new_patch.shape: ', new_patch.shape)

            updated_patch = np.maximum(old_patch, new_patch)
            mask_stitched[start_h:end_h, start_w:end_w] = updated_patch[:, :]

        stitched_mask_list.append(mask_stitched)

        save_path = base_mask_path.replace(pred_mask_folder, stitched_folder)
        cv2.imwrite(save_path, mask_stitched)

        # save a colored version for visualization
        #print('mask_stitched.shape: ', mask_stitched.shape)
        colored_mask_stitched = np.zeros((mask_stitched.shape[0], mask_stitched.shape[1], 3), dtype=np.uint8)
        colored_mask_stitched[mask_stitched == 1] = (0, 255, 0)
        cv2.imwrite(save_path.replace(pred_mask_folder, colored_stitched_folder), colored_mask_stitched)
    
    log(f'Done stitching {len(mask_list)} patches. Stitched: {len(stitched_mask_list)}.')

    return stitched_mask_list, stitched_folder

import utils.metrics as metrics
from preprocessing.Metas import Organ2FileID
def eval_stitched(pred_folder, true_folder, organ='Colon') -> dict:
    '''
        Evaluation on final stitched mask against the ground truth mask.
    
    '''
    pred_list = sorted(glob(os.path.join(pred_folder + '/*.png')))
    true_list = sorted(glob(os.path.join(true_folder + '/*.png')))
    # Filter out other organs
    file_ids = Organ2FileID[organ]['test']
    true_list = [x for x in true_list if any([f'{file_id}' in x for file_id in file_ids])]
    print('pred_folder: ', pred_folder, '\ntrue_folder: ', true_folder)
    print(len(pred_list), len(true_list))

    assert len(pred_list) == len(true_list)

    metric_list = []
    for pred_mask_path, true_mask_path in zip(pred_list, true_list):
        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
        true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)
        assert pred_mask.shape == true_mask.shape

        metric = metrics.compute_metrics(pred_mask, true_mask, ['p_F1', 'aji', 'iou'])
        metric_list.append(metric)

    eval_results = {}
    for key in metric_list[0].keys():
        num = sum([i[key] for i in metric_list]) / len(metric_list)
        eval_results[key] = num
        print(F'{key}: {num}')
    
    return eval_results

    

@torch.no_grad()
def infer(config, wandb_run=None):
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

    # Set all the paths.
    model_name = f'{config.percentage:.3f}_{config.organ}_m{config.multiplier}_MoNuSeg_depth{config.depth}_seed{config.random_seed}_{config.latent_loss}'
    config.matched_pair_path_root = os.path.join(config.output_save_root, model_name)
    config.model_save_path = os.path.join(config.output_save_root, model_name, 'reg2seg.ckpt')
    config.log_dir = os.path.join(config.log_folder, model_name) # This is log file path.
    save_folder = os.path.join(config.output_save_root, model_name, 'reg2seg')
    pred_mask_folder = os.path.join(config.output_save_root, model_name, 'pred_patches')
    os.makedirs(pred_mask_folder, exist_ok=True)

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
    load_results = load_match_pairs(config.matched_pair_path_root, mode='infer', config=config)
    (test_images, test_masks, closest_images, closest_masks, test_image_paths) = load_results
    dataset = torch.utils.data.TensorDataset(closest_images, test_images, closest_masks, test_masks)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Step 2: Predict & Apply the warping field.
    warp_predictor.eval()
    pred_mask_list = []
    print(f'Starting inference for {test_images.shape[0]} test images ...')
    for iter_idx, (bclosest_images, btest_images, bclosest_masks, btest_masks) in enumerate(tqdm(dataloader)):
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
        images_U2A = warper(btest_images, flow=warp_field_forward)
        images_U2A_A2U = warper(images_U2A, flow=warp_field_reverse)
        pred_masks = warper(bclosest_masks, flow=warp_field_reverse)
        bpred_mask_list = [m.cpu().detach().numpy() for m in pred_masks]
        pred_mask_list.extend(bpred_mask_list)
        
        plot_freq = 10
        shall_plot = iter_idx % plot_freq == 0
        #print('shall_plot: ', shall_plot, iter_idx, plot_freq, len(dataset))
        if shall_plot:
            save_path_fig_sbs = '%s/infer/figure_sample%s.png' % (
                save_folder, str(iter_idx).zfill(5))
            plot_side_by_side(save_path_fig_sbs, *numpy_variables(
                btest_images[0], bclosest_images[0],
                images_U2A_A2U[0], images_U2A[0],
                btest_masks[0] > 0.5, bclosest_masks[0] > 0.5,
                pred_masks[0] > 0.5))

    assert len(pred_mask_list) == len(test_masks)
    print('Completed inference.')
    print('len(test_masks), len(pred_mask_list): ', len(test_masks), len(pred_mask_list))
    
    # Step 3: Evaluation. Compute Dice Coeff.
    print(f'Computing Dice Coeff for {len(pred_mask_list)} total masks...')
    dice_list = []
    iou_list = []

    # convert between torch Tensor & np array
    if 'torch' in str(type(test_masks)):
        test_masks = test_masks.cpu().detach().numpy()

    for i in range(len(pred_mask_list)):
        # print('test_masks[i].shape, pred_mask_list[i].shape: ', \
        #       test_masks[i].shape, pred_mask_list[i].shape)
        dice_list.append(
            dice_coeff((np.expand_dims(test_masks[i], 0) > 0.5).transpose(1, 2, 0),
                       (pred_mask_list[i] > 0.5).transpose(1, 2, 0)))
        iou_list.append(
            IoU((np.expand_dims(test_masks[i], 0) > 0.5).transpose(1, 2, 0),
                       (pred_mask_list[i] > 0.5).transpose(1, 2, 0)))
        
        # save to disk
        fname = os.path.join(pred_mask_folder, os.path.basename(test_image_paths[i]))
        print((pred_mask_list[i] > 0.5).astype(np.uint8).shape)
        cv2.imwrite(fname, np.squeeze((pred_mask_list[i] > 0.5).astype(np.uint8)))
    
    # Stitch the masks together.
    stitched_mask_list, stitched_folder = stitch_patches(pred_mask_folder)
    test_mask_folder = config.groudtruth_folder
    stitched_results = eval_stitched(stitched_folder, test_mask_folder, organ=config.organ)
    
    log('[Eval] Dice coeff (seg): %.3f \u00B1 %.3f.'
        % (np.mean(dice_list), np.std(dice_list)),
        filepath=config.log_dir,
        to_console=True)
    log('[Eval] IoU (seg): %.3f \u00B1 %.3f.'
        % (np.mean(iou_list), np.std(iou_list)),
        filepath=config.log_dir,
        to_console=True)
    
    for k, v in stitched_results.items():
        log(F'[Eval] Stitched {k}: {v}', filepath=config.log_dir, to_console=True) 

    if wandb_run is not None:
        wandb_run.log({'infer/dice_seg_mean': np.mean(dice_list),
                       'infer/dice_seg_std': np.std(dice_list),
                       'infer/iou_seg_mean': np.mean(iou_list),
                       'infer/iou_seg_std': np.std(iou_list)})
        for k, v in stitched_results.items():
            wandb_run.log({F'infer/stitched_{k}': v})

    return

from dotenv import load_dotenv

load_dotenv('../.env')
WANDB_ENTITY = os.getenv('WANDB_ENTITY')
PROJECT_PATH = os.getenv('PROJECT_PATH')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--mode', help='`train` or `test` or `infer` ?', default='train')
    parser.add_argument('--model_config', help='Path to model config file', default='./config/MoNuSeg_reg2seg.yaml')
    parser.add_argument('--data_config', help='Path to data config file', default='./config/MoNuSeg_data.yaml')
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)
    parser.add_argument('--num-workers', help='Number of workers, e.g. use number of cores', default=4, type=int)
    parser.add_argument('--aiae-depth', help='Depth of the AIAE model', default=5, type=int) # in sync with aiae model
    parser.add_argument('--aiae-latent-loss', help='latent loss of AIAE model', default='SimCLR', type=str) #  in sync with aiae model
    args = parser.parse_args()

    model_config = OmegaConf.load(args.model_config)
    data_config = OmegaConf.load(args.data_config)
    config = OmegaConf.merge(model_config, data_config)
    
    config.gpu_id = args.gpu_id
    config.num_workers = args.num_workers
    config.depth = args.aiae_depth
    config.latent_loss = args.aiae_latent_loss

    print(config)
    seed_everything(config.random_seed)

    assert args.mode in ['train', 'test', 'infer']

    wandb_run = None
    import wandb
    if config.use_wandb and args.mode == 'train':
        wandb_run = wandb.init(
            entity=WANDB_ENTITY,
            project="cellseg",
            name="monuseg-reg2seg",
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
            reinit=True,
            settings=wandb.Settings(start_method="thread")
        )

    if args.mode == 'train':
        train(config=config, wandb_run=wandb_run)
        #test(config=config)
        infer(config=config, wandb_run=wandb_run)
    elif args.mode == 'test':
        pass
        #test(config=config)
    elif args.mode == 'infer':
        infer(config=config, wandb_run=wandb_run)

    if wandb_run is not None:
        wandb_run.finish()