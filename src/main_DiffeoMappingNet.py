from typing import Tuple, List, Literal
import argparse
import numpy as np
import torch
import cv2
import ast
import os
# from model.scheduler import LinearWarmupCosineAnnealingLR
from tqdm import tqdm
from matplotlib import pyplot as plt
from functools import partial
import wandb

from registration.registration_loss import GradLoss as SmoothnessLoss
from registration.spatial_transformer import SpatialTransformer as Warper
from registration.unet import UNet
from registration.voxelmorph import VxmDense as VoxelMorph
from registration.corrmlp import CorrMLP

from utils.attribute_hashmap import AttributeHashmap
from utils.prepare_dataset import prepare_dataset
from utils.log_util import log
from utils.parse import parse_settings
from utils.seed import seed_everything
from utils.early_stop import EarlyStopping
from utils.metrics import dice_coeff, IoU, l1
from utils.ncc import NCC


def build_diffeomappingnet(config):
    if config.DiffeoMappingNet_model == 'UNet':
        warp_predictor = UNet(
            num_filters=config.num_filters,
            in_channels=6,
            out_channels=4,
        )
    elif config.DiffeoMappingNet_model == 'VM':
        warp_predictor = VoxelMorph(
            inshape=(32, 32),
            src_feats=3,
            trg_feats=3,
            int_steps=0,  # non-diffeomorphic
            bidir=True,
        )
    elif config.DiffeoMappingNet_model == 'VM-Diff':
        warp_predictor = VoxelMorph(
            inshape=(32, 32),
            src_feats=3,
            trg_feats=3,
            bidir=True,
        )
    elif config.DiffeoMappingNet_model == 'CorrMLP':
        # NOTE: `CorrMLP` needs smaller learning rate (recommended: 1e-4).
        warp_predictor = CorrMLP(
            in_channels=3,
            enc_channels=config.num_filters,
        )
    else:
        raise ValueError('`config.DiffeoMappingNet_model`: %s not supported.' % config.DiffeoMappingNet_model)
    return warp_predictor

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

def direction_from_gradient(gradient_image: np.array, thr: float = 0) -> float:
    '''
    Find the main axis direction from a gradient image.
    '''
    # Compute the gradient.
    if gradient_image.shape[-1] == 1:
        gradient_image = gradient_image.squeeze()
    gradient_x, gradient_y = np.gradient(gradient_image)

    # Mask out the zero values.
    mask = gradient_image > thr
    gradient_x = np.where(mask, gradient_x, 0)
    gradient_y = np.where(mask, gradient_y, 0)

    # Calculate the gradient direction.
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    # Calculate the average direction.
    avg_direction = np.arctan2(np.mean(np.sin(gradient_direction[mask])), np.mean(np.cos(gradient_direction[mask])))
    # Convert the average direction from radians to degrees.
    avg_direction_degrees = np.degrees(avg_direction)

    return avg_direction_degrees

def angular_diff(im1, im2) -> float:
    direction1_deg = direction_from_gradient(im1, thr=-0.8)
    direction2_deg = direction_from_gradient(im2, thr=-0.8)
    difference = np.abs(direction1_deg - direction2_deg)
    difference = np.where(difference > 180, 360 - difference, difference)
    return difference

def plot_side_by_side(save_path,
                      im_U, im_A,
                      im_U2A_A2U, im_U2A,
                      ma_U, ma_A, ma_A2U,
                      metric_name_list, metric_list):
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
    ax.set_title('Unannotated label (U)')
    ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, 4, 4)
    ax.imshow(np.clip(ma_A, 0, 1), cmap='gray')
    ax.set_title('Annotated label (A)')
    ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, 4, 7)
    ax.imshow(np.clip(ma_A2U, 0, 1), cmap='gray')
    ax.set_title('Projected label (A->U)')
    ax.set_axis_off()

    metric_str = ''
    for metric_name, metric in zip(metric_name_list, metric_list):
        metric_str += '%s (label(U), label(A)) = %.3f, %s (label(U), label(A->U)) = %.3f\n' % (
        metric_name, metric(ma_U, ma_A), metric_name, metric(ma_U, ma_A2U))
    fig_sbs.suptitle(metric_str, fontsize=15)
    fig_sbs.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig_sbs.savefig(save_path)
    plt.close(fig=fig_sbs)

    return

def flip_rot(image, fliplr: bool, rot_angle: Literal[0, 90, 180, 270]):
    if fliplr:
        image = torch.flip(image, dims=(2,))

    if rot_angle == 90:
        image = torch.rot90(image, k=1, dims=(2, 3))
    elif rot_angle == 180:
        image = torch.rot90(image, k=2, dims=(2, 3))
    elif rot_angle == 270:
        image = torch.rot90(image, k=3, dims=(2, 3))
    else:
        assert rot_angle == 0

    return image

def predict_flipping_rotation(image_fixed: torch.Tensor, image_moving: torch.Tensor):
    '''
    Predict the flipping and rotation transformation:
    image_fixed \approx flipping_rotation_transform(image_moving).

    This function does not perform deep learning.
    It's purely iterating the 2x4=8 possibilities.

    This function assumes the images are of shape [batch_size, C, H, W]
    '''

    assert image_fixed.shape == image_moving.shape

    cross_corr_op = NCC(image_fixed)
    cross_corr_list = []
    transform_forward_list, transform_reverse_list = [], []

    for fliplr_forward, fliplr_reverse in zip([False, True], [False, True]):
        for rot_angle_forward, rot_angle_reverse in \
            zip([0, 90, 180, 270], [0, 270, 180, 90]):

            transform_func_forward = partial(flip_rot, fliplr=fliplr_forward, rot_angle=rot_angle_forward)
            transform_func_reverse = partial(flip_rot, fliplr=fliplr_reverse, rot_angle=rot_angle_reverse)

            transform_forward_list.append(transform_func_forward)
            transform_reverse_list.append(transform_func_reverse)
            transformed_image = transform_func_forward(image_moving)

            cross_corr_list.append(torch.max(torch.max(cross_corr_op(transformed_image), dim=1)[0], dim=1)[0])

    cross_corr = torch.stack(cross_corr_list)
    transform_arr_forward = np.array(transform_forward_list)
    transform_arr_reverse = np.array(transform_reverse_list)

    # Use cross correlation to decide which transformed image is the best match.
    best_transform_indices = cross_corr.argmax(0).cpu().numpy()

    flip_rot_transform_forward = transform_arr_forward[best_transform_indices]
    flip_rot_transform_reverse = transform_arr_reverse[best_transform_indices]

    return flip_rot_transform_forward, flip_rot_transform_reverse

def apply_flipping_rotation(transform, image):
    transformed_image = torch.zeros_like(image)
    for batch_idx in range(len(transform)):
        transformed_image[batch_idx] = transform[batch_idx](image[batch_idx].unsqueeze(0)).squeeze(0)
    return transformed_image

# FIXME!: I think we can even use another cycle loss: AM -> UM -> AM
# FIXME!: Also, if the augmented label is good, we can use it as a target for the forward cycle: UM -> AM.
def train(config, wandb_run=None):

    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')

    dataset, train_loader, val_loader, test_loader = prepare_dataset(config=config)

    # Build the model
    warp_predictor = build_diffeomappingnet(config)
    warp_predictor = warp_predictor.to(device)

    warper = Warper(size=config.target_dim)
    warper = warper.to(device)

    optimizer = torch.optim.AdamW(warp_predictor.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    loss_fn_mse = torch.nn.MSELoss()
    loss_fn_smoothness = SmoothnessLoss('l2').loss
    early_stopper = EarlyStopping(mode='min',
                                  patience=config.patience,
                                  percentage=False)

    best_val_loss = np.inf
    for epoch_idx in tqdm(range(config.max_epochs)):
        dataset.set_deterministic(False)
        train_loss, train_loss_forward, train_loss_cyclic = 0, 0, 0
        train_metric_ref_dict, train_metric_fliprot_dict, train_metric_ours_dict = {}, {}, {}

        warp_predictor.train()
        plot_freq = int(len(train_loader) // config.n_plot_per_epoch)
        for iter_idx, (_, _, image_n_view, label_n_view, _, _, _) in enumerate(tqdm(train_loader)):

            assert image_n_view.shape[1] == 2
            curr_batch_size = image_n_view.shape[0]
            shall_plot = iter_idx % plot_freq == plot_freq - 1

            if config.hard_example_ratio == 0:
                # No hard example: map within the same cell (one augmented version to another).
                unannotated_images = image_n_view[:, 0, ...]
                unannotated_labels = label_n_view[:, 0, ...]
                annotated_images = image_n_view[:, 1, ...]
                annotated_labels = label_n_view[:, 1, ...]

            else:
                # Hard example mining: map across different cells.
                num_permuted = int(len(image_n_view) * config.hard_example_ratio)
                permuted_idx = np.random.permutation(num_permuted)
                unannotated_images = image_n_view[:, 0, ...]
                unannotated_labels = label_n_view[:, 0, ...]
                permuted_image_n_view = image_n_view.clone()
                permuted_label_n_view = label_n_view.clone()
                permuted_image_n_view[:num_permuted] = image_n_view[:num_permuted][permuted_idx]
                permuted_label_n_view[:num_permuted] = label_n_view[:num_permuted][permuted_idx]
                annotated_images = permuted_image_n_view[:, 1, ...]
                annotated_labels = permuted_label_n_view[:, 1, ...]

            if len(unannotated_labels.shape) == 3:
                unannotated_labels = unannotated_labels[:, None, ...]
            if len(annotated_labels.shape) == 3:
                annotated_labels = annotated_labels[:, None, ...]
            label_is_binary = not torch.is_floating_point(annotated_labels)

            unannotated_images = unannotated_images.float().to(device) # (bsz, in_chan, H, W)
            annotated_images = annotated_images.float().to(device)
            unannotated_labels = unannotated_labels.float().to(device)
            annotated_labels = annotated_labels.float().to(device)

            if label_is_binary:
                # Only care about the binary label.
                assert annotated_labels.max() in [0, 1]
                annotated_labels = (annotated_labels > 0.5).float()
                unannotated_labels = (unannotated_labels > 0.5).float()
            else:
                annotated_labels = annotated_labels.float()
                unannotated_labels = unannotated_labels.float()

            # Predict flipping and rotation, and apply correction.
            flip_rot_transform_forward, _ = predict_flipping_rotation(unannotated_images, annotated_images)
            annotated_images_fliprot = apply_flipping_rotation(flip_rot_transform_forward, annotated_images)
            annotated_labels_fliprot = apply_flipping_rotation(flip_rot_transform_forward, annotated_labels)

            warp_field_forward, warp_field_reverse = warp_predictor(source=unannotated_images, target=annotated_images_fliprot)

            # Apply the warping field.
            images_U2A = warper(unannotated_images, flow=warp_field_forward)
            images_U2A_A2U = warper(images_U2A, flow=warp_field_reverse)
            labels_A2U = warper(annotated_labels_fliprot, flow=warp_field_reverse)

            if label_is_binary:
                labels_A2U = (labels_A2U > 0.5).float()

            # Compute metric.
            if label_is_binary:
                metric_name_list = ['DSC', 'IOU']
                metric_list = [dice_coeff, IoU]
            else:
                metric_name_list = ['L1', 'AngularDiff']
                metric_list = [l1, angular_diff]

            for i in range(len(labels_A2U)):
                for metric_name, metric in zip(metric_name_list, metric_list):
                    if metric_name not in train_metric_ref_dict.keys():
                        train_metric_ref_dict[metric_name] = []
                        train_metric_fliprot_dict[metric_name] = []
                        train_metric_ours_dict[metric_name] = []

                    train_metric_ref_dict[metric_name].append(
                        metric((annotated_labels[i, ...]).cpu().detach().numpy().transpose(1, 2, 0),
                            (unannotated_labels[i, ...]).cpu().detach().numpy().transpose(1, 2, 0)))
                    train_metric_fliprot_dict[metric_name].append(
                        metric((annotated_labels_fliprot[i, ...]).cpu().detach().numpy().transpose(1, 2, 0),
                            (unannotated_labels[i, ...]).cpu().detach().numpy().transpose(1, 2, 0)))
                    train_metric_ours_dict[metric_name].append(
                        metric((labels_A2U[i, ...]).cpu().detach().numpy().transpose(1, 2, 0),
                            (unannotated_labels[i, ...]).cpu().detach().numpy().transpose(1, 2, 0)))

            if shall_plot:
                save_path_fig_sbs = '%s/train/figure_log_epoch%s_sample%s.png' % (
                    config.output_save_path, str(epoch_idx).zfill(5), str(iter_idx).zfill(5))

                if label_is_binary:
                    plot_side_by_side(save_path_fig_sbs, *numpy_variables(
                        unannotated_images[0], annotated_images[0],
                        images_U2A_A2U[0], images_U2A[0],
                        unannotated_labels[0] > 0.5, annotated_labels[0] > 0.5,
                        labels_A2U[0] > 0.5), metric_name_list, metric_list)
                else:
                    plot_side_by_side(save_path_fig_sbs, *numpy_variables(
                        unannotated_images[0], annotated_images[0],
                        images_U2A_A2U[0], images_U2A[0],
                        unannotated_labels[0], annotated_labels[0],
                        labels_A2U[0]), metric_name_list, metric_list)

            loss_forward = loss_fn_mse(annotated_images, images_U2A)
            loss_cyclic = loss_fn_mse(unannotated_images, images_U2A_A2U)
            loss = loss_forward + loss_cyclic

            if config.coeff_smoothness:
                loss_smoothness = loss_fn_smoothness(np.zeros((1)), warp_field_forward) \
                                + loss_fn_smoothness(np.zeros((1)), warp_field_reverse)
                loss += config.coeff_smoothness * loss_smoothness

            train_loss += loss.item() * curr_batch_size
            train_loss_forward += loss_forward.item() * curr_batch_size
            train_loss_cyclic += loss_cyclic.item() * curr_batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader.dataset)
        train_loss_forward /= len(train_loader.dataset)
        train_loss_cyclic /= len(train_loader.dataset)

        lr_scheduler.step()

        log_str = 'Train [%s/%s] loss: %.3f, forward: %.3f, cyclic: %.3f. ' % (
            epoch_idx + 1, config.max_epochs, train_loss, train_loss_forward, train_loss_cyclic)

        for metric_name in metric_name_list:
            log_str += '%s (ref): %.3f \u00B1 %.3f, %s (flip+rot): %.3f \u00B1 %.3f, %s (ours): %.3f \u00B1 %.3f. ' % (
                metric_name, np.mean(train_metric_ref_dict[metric_name]), np.std(train_metric_ref_dict[metric_name]),
                metric_name, np.mean(train_metric_fliprot_dict[metric_name]), np.std(train_metric_fliprot_dict[metric_name]),
                metric_name, np.mean(train_metric_ours_dict[metric_name]), np.std(train_metric_ours_dict[metric_name]),
            )

        log(log_str,
            filepath=config.log_path,
            to_console=False)

        if wandb_run is not None:
            log_dict = {
                'train/loss': train_loss,
                'train/loss_forward': train_loss_forward,
                'train/loss_cyclic': train_loss_cyclic,
            }
            for metric_name in metric_name_list:
                log_dict[f'train/{metric_name}_ref_mean'] = np.mean(train_metric_ref_dict[metric_name])
                log_dict[f'train/{metric_name}_ref_std'] = np.std(train_metric_ref_dict[metric_name])
                log_dict[f'train/{metric_name}_fliprot_mean'] = np.mean(train_metric_fliprot_dict[metric_name])
                log_dict[f'train/{metric_name}_fliprot_std'] = np.std(train_metric_fliprot_dict[metric_name])
                log_dict[f'train/{metric_name}_ours_mean'] = np.mean(train_metric_ours_dict[metric_name])
                log_dict[f'train/{metric_name}_ours_std'] = np.std(train_metric_ours_dict[metric_name])
            wandb_run.log(log_dict)

        # Validation.
        dataset.set_deterministic(True)
        warp_predictor.eval()
        with torch.no_grad():
            val_loss, val_loss_forward, val_loss_cyclic = 0, 0, 0
            val_metric_ref_dict, val_metric_fliprot_dict, val_metric_ours_dict = {}, {}, {}

            plot_freq = int(len(val_loader) // config.n_plot_per_epoch)
            for iter_idx, (_, _, image_n_view, label_n_view, _, _, _) in enumerate(tqdm(val_loader)):

                assert image_n_view.shape[1] == 2
                curr_batch_size = image_n_view.shape[0]
                shall_plot = iter_idx % plot_freq == plot_freq - 1

                # Hard example mining only relevant to training.
                # For val and test we always use the same cell.
                unannotated_images = image_n_view[:, 0, ...]
                unannotated_labels = label_n_view[:, 0, ...]
                annotated_images = image_n_view[:, 1, ...]
                annotated_labels = label_n_view[:, 1, ...]

                if len(unannotated_labels.shape) == 3:
                    unannotated_labels = unannotated_labels[:, None, ...]
                if len(annotated_labels.shape) == 3:
                    annotated_labels = annotated_labels[:, None, ...]
                label_is_binary = not torch.is_floating_point(annotated_labels)

                unannotated_images = unannotated_images.float().to(device) # (bsz, in_chan, H, W)
                annotated_images = annotated_images.float().to(device)
                unannotated_labels = unannotated_labels.float().to(device)
                annotated_labels = annotated_labels.float().to(device)

                if label_is_binary:
                    # Only care about the binary label.
                    assert annotated_labels.max() in [0, 1]
                    annotated_labels = (annotated_labels > 0.5).float()
                    unannotated_labels = (unannotated_labels > 0.5).float()
                else:
                    annotated_labels = annotated_labels.float()
                    unannotated_labels = unannotated_labels.float()

                # Predict flipping and rotation, and apply correction.
                flip_rot_transform_forward, _ = predict_flipping_rotation(unannotated_images, annotated_images)
                annotated_images_fliprot = apply_flipping_rotation(flip_rot_transform_forward, annotated_images)
                annotated_labels_fliprot = apply_flipping_rotation(flip_rot_transform_forward, annotated_labels)

                warp_field_forward, warp_field_reverse = warp_predictor(source=unannotated_images, target=annotated_images_fliprot)

                # Apply the warping field.
                images_U2A = warper(unannotated_images, flow=warp_field_forward)
                images_U2A_A2U = warper(images_U2A, flow=warp_field_reverse)
                labels_A2U = warper(annotated_labels_fliprot, flow=warp_field_reverse)

                if label_is_binary:
                    labels_A2U = (labels_A2U > 0.5).float()

                # Compute metric.
                if label_is_binary:
                    metric_name_list = ['DSC', 'IOU']
                    metric_list = [dice_coeff, IoU]
                else:
                    metric_name_list = ['L1', 'AngularDiff']
                    metric_list = [l1, angular_diff]

                for i in range(len(labels_A2U)):
                    for metric_name, metric in zip(metric_name_list, metric_list):
                        if metric_name not in val_metric_ref_dict.keys():
                            val_metric_ref_dict[metric_name] = []
                            val_metric_fliprot_dict[metric_name] = []
                            val_metric_ours_dict[metric_name] = []

                        val_metric_ref_dict[metric_name].append(
                            metric((annotated_labels[i, ...]).cpu().detach().numpy().transpose(1, 2, 0),
                                   (unannotated_labels[i, ...]).cpu().detach().numpy().transpose(1, 2, 0)))
                        val_metric_fliprot_dict[metric_name].append(
                            metric((annotated_labels_fliprot[i, ...]).cpu().detach().numpy().transpose(1, 2, 0),
                                   (unannotated_labels[i, ...]).cpu().detach().numpy().transpose(1, 2, 0)))
                        val_metric_ours_dict[metric_name].append(
                            metric((labels_A2U[i, ...]).cpu().detach().numpy().transpose(1, 2, 0),
                                   (unannotated_labels[i, ...]).cpu().detach().numpy().transpose(1, 2, 0)))

                if shall_plot:
                    save_path_fig_sbs = '%s/val/figure_log_epoch%s_sample%s.png' % (
                        config.output_save_path, str(epoch_idx).zfill(5), str(iter_idx).zfill(5))

                    if label_is_binary:
                        plot_side_by_side(save_path_fig_sbs, *numpy_variables(
                            unannotated_images[0], annotated_images[0],
                            images_U2A_A2U[0], images_U2A[0],
                            unannotated_labels[0] > 0.5, annotated_labels[0] > 0.5,
                            labels_A2U[0] > 0.5), metric_name_list, metric_list)
                    else:
                        plot_side_by_side(save_path_fig_sbs, *numpy_variables(
                            unannotated_images[0], annotated_images[0],
                            images_U2A_A2U[0], images_U2A[0],
                            unannotated_labels[0], annotated_labels[0],
                            labels_A2U[0]), metric_name_list, metric_list)

                loss_forward = loss_fn_mse(annotated_images, images_U2A)
                loss_cyclic = loss_fn_mse(unannotated_images, images_U2A_A2U)
                loss = loss_forward + loss_cyclic
                if config.coeff_smoothness:
                    loss_smoothness = loss_fn_smoothness(np.zeros((1)), warp_field_forward) \
                                    + loss_fn_smoothness(np.zeros((1)), warp_field_reverse)
                    loss += config.coeff_smoothness * loss_smoothness

                val_loss += loss.item() * curr_batch_size
                val_loss_forward += loss_forward.item() * curr_batch_size
                val_loss_cyclic += loss_cyclic.item() * curr_batch_size

        val_loss /= len(val_loader.dataset)
        val_loss_forward /= len(val_loader.dataset)
        val_loss_cyclic /= len(val_loader.dataset)

        log_str = 'Validation [%s/%s] loss: %.3f, forward: %.3f, cyclic: %.3f. ' % (
            epoch_idx + 1, config.max_epochs, val_loss, val_loss_forward, val_loss_cyclic)

        for metric_name in metric_name_list:
            log_str += '%s (ref): %.3f \u00B1 %.3f, %s (flip+rot): %.3f \u00B1 %.3f, %s (ours): %.3f \u00B1 %.3f. ' % (
                metric_name, np.mean(val_metric_ref_dict[metric_name]), np.std(val_metric_ref_dict[metric_name]),
                metric_name, np.mean(val_metric_fliprot_dict[metric_name]), np.std(val_metric_fliprot_dict[metric_name]),
                metric_name, np.mean(val_metric_ours_dict[metric_name]), np.std(val_metric_ours_dict[metric_name]),
            )

        log(log_str,
            filepath=config.log_path,
            to_console=False)

        if wandb_run is not None:
            log_dict = {
                'val/loss': val_loss,
                'val/loss_forward': val_loss_forward,
                'val/loss_cyclic': val_loss_cyclic
            }
            for metric_name in metric_name_list:
                log_dict[f'val/{metric_name}_ref_mean'] = np.mean(val_metric_ref_dict[metric_name])
                log_dict[f'val/{metric_name}_ref_std'] = np.std(val_metric_ref_dict[metric_name])
                log_dict[f'val/{metric_name}_fliprot_mean'] = np.mean(val_metric_fliprot_dict[metric_name])
                log_dict[f'val/{metric_name}_fliprot_std'] = np.std(val_metric_fliprot_dict[metric_name])
                log_dict[f'val/{metric_name}_ours_mean'] = np.mean(val_metric_ours_dict[metric_name])
                log_dict[f'val/{metric_name}_ours_std'] = np.std(val_metric_ours_dict[metric_name])
            wandb_run.log(log_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            warp_predictor.save_weights(config.DiffeoMappingNet_model_save_path)
            log('%s: Model weights successfully saved.' % config.DiffeoMappingNet_model,
                filepath=config.log_path,
                to_console=False)

        if early_stopper.step(val_loss):
            log('Early stopping criterion met. Ending training.',
                filepath=config.log_path,
                to_console=True)
            break

    return


@torch.no_grad()
def test(config: AttributeHashmap, n_plot_per_epoch: int = None):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    _, _, _, test_set = prepare_dataset(config=config)

    dataset, train_loader, val_loader, test_loader = prepare_dataset(config=config)

    # Build the model
    warp_predictor = build_diffeomappingnet(config)
    warp_predictor.load_weights(config.DiffeoMappingNet_model_save_path, device=device)
    warp_predictor = warp_predictor.to(device)

    warper = Warper(size=config.target_dim)
    warper = warper.to(device)

    loss_fn_mse = torch.nn.MSELoss()
    loss_fn_smoothness = SmoothnessLoss('l2').loss

    dataset.set_deterministic(True)
    warp_predictor.eval()
    if n_plot_per_epoch is not None:
        plot_freq = int(len(test_set) // n_plot_per_epoch)
    else:
        plot_freq = 1

    test_loss, test_loss_forward, test_loss_cyclic = 0, 0, 0
    test_metric_ref_dict, test_metric_fliprot_dict, test_metric_ours_dict = {}, {}, {}

    plot_freq = int(len(test_loader) // config.n_plot_per_epoch)
    for iter_idx, (_, _, image_n_view, label_n_view, _, _, _) in enumerate(tqdm(test_loader)):

        assert image_n_view.shape[1] == 2
        curr_batch_size = image_n_view.shape[0]
        shall_plot = iter_idx % plot_freq == plot_freq - 1

        # Hard example mining only relevant to training.
        # For val and test we always use the same cell.
        unannotated_images = image_n_view[:, 0, ...]
        unannotated_labels = label_n_view[:, 0, ...]
        annotated_images = image_n_view[:, 1, ...]
        annotated_labels = label_n_view[:, 1, ...]

        if len(unannotated_labels.shape) == 3:
            unannotated_labels = unannotated_labels[:, None, ...]
        if len(annotated_labels.shape) == 3:
            annotated_labels = annotated_labels[:, None, ...]
        label_is_binary = not torch.is_floating_point(annotated_labels)

        unannotated_images = unannotated_images.float().to(device) # (bsz, in_chan, H, W)
        annotated_images = annotated_images.float().to(device)
        unannotated_labels = unannotated_labels.float().to(device)
        annotated_labels = annotated_labels.float().to(device)

        if label_is_binary:
            # Only care about the binary label.
            assert annotated_labels.max() in [0, 1]
            annotated_labels = (annotated_labels > 0.5).float()
            unannotated_labels = (unannotated_labels > 0.5).float()
        else:
            annotated_labels = annotated_labels.float()
            unannotated_labels = unannotated_labels.float()

        # Predict flipping and rotation, and apply correction.
        flip_rot_transform_forward, _ = predict_flipping_rotation(unannotated_images, annotated_images)
        annotated_images_fliprot = apply_flipping_rotation(flip_rot_transform_forward, annotated_images)
        annotated_labels_fliprot = apply_flipping_rotation(flip_rot_transform_forward, annotated_labels)

        warp_field_forward, warp_field_reverse = warp_predictor(source=unannotated_images, target=annotated_images_fliprot)

        # Apply the warping field.
        images_U2A = warper(unannotated_images, flow=warp_field_forward)
        images_U2A_A2U = warper(images_U2A, flow=warp_field_reverse)
        labels_A2U = warper(annotated_labels_fliprot, flow=warp_field_reverse)

        if label_is_binary:
            labels_A2U = (labels_A2U > 0.5).float()

        # Compute metric.
        if label_is_binary:
            metric_name_list = ['DSC', 'IOU']
            metric_list = [dice_coeff, IoU]
        else:
            metric_name_list = ['L1', 'AngularDiff']
            metric_list = [l1, angular_diff]

        for i in range(len(labels_A2U)):
            for metric_name, metric in zip(metric_name_list, metric_list):
                if metric_name not in test_metric_ref_dict.keys():
                    test_metric_ref_dict[metric_name] = []
                    test_metric_fliprot_dict[metric_name] = []
                    test_metric_ours_dict[metric_name] = []

                test_metric_ref_dict[metric_name].append(
                    metric((annotated_labels[i, ...]).cpu().detach().numpy().transpose(1, 2, 0),
                           (unannotated_labels[i, ...]).cpu().detach().numpy().transpose(1, 2, 0)))
                test_metric_fliprot_dict[metric_name].append(
                    metric((annotated_labels_fliprot[i, ...]).cpu().detach().numpy().transpose(1, 2, 0),
                           (unannotated_labels[i, ...]).cpu().detach().numpy().transpose(1, 2, 0)))
                test_metric_ours_dict[metric_name].append(
                    metric((labels_A2U[i, ...]).cpu().detach().numpy().transpose(1, 2, 0),
                           (unannotated_labels[i, ...]).cpu().detach().numpy().transpose(1, 2, 0)))

        if shall_plot:
            save_path_fig_sbs = '%s/test/figure_log_sample%s.png' % (
                config.output_save_path, str(iter_idx).zfill(5))

            if label_is_binary:
                plot_side_by_side(save_path_fig_sbs, *numpy_variables(
                    unannotated_images[0], annotated_images[0],
                    images_U2A_A2U[0], images_U2A[0],
                    unannotated_labels[0] > 0.5, annotated_labels[0] > 0.5,
                    labels_A2U[0] > 0.5), metric_name_list, metric_list)
            else:
                plot_side_by_side(save_path_fig_sbs, *numpy_variables(
                    unannotated_images[0], annotated_images[0],
                    images_U2A_A2U[0], images_U2A[0],
                    unannotated_labels[0], annotated_labels[0],
                    labels_A2U[0]), metric_name_list, metric_list)

        loss_forward = loss_fn_mse(annotated_images, images_U2A)
        loss_cyclic = loss_fn_mse(unannotated_images, images_U2A_A2U)
        loss = loss_forward + loss_cyclic
        if config.coeff_smoothness:
            loss_smoothness = loss_fn_smoothness(np.zeros((1)), warp_field_forward) \
                            + loss_fn_smoothness(np.zeros((1)), warp_field_reverse)
            loss += config.coeff_smoothness * loss_smoothness

        test_loss += loss.item() * curr_batch_size
        test_loss_forward += loss_forward.item() * curr_batch_size
        test_loss_cyclic += loss_cyclic.item() * curr_batch_size

    test_loss /= len(test_loader.dataset)
    test_loss_forward /= len(test_loader.dataset)
    test_loss_cyclic /= len(test_loader.dataset)

    log_str = 'Test loss: %.3f, forward: %.3f, cyclic: %.3f. ' % (
        test_loss, test_loss_forward, test_loss_cyclic)

    for metric_name in metric_name_list:
        log_str += '%s (ref): %.3f \u00B1 %.3f, %s (flip+rot): %.3f \u00B1 %.3f, %s (ours): %.3f \u00B1 %.3f. ' % (
            metric_name, np.mean(test_metric_ref_dict[metric_name]), np.std(test_metric_ref_dict[metric_name]),
            metric_name, np.mean(test_metric_fliprot_dict[metric_name]), np.std(test_metric_fliprot_dict[metric_name]),
            metric_name, np.mean(test_metric_ours_dict[metric_name]), np.std(test_metric_ours_dict[metric_name]),
        )

    log(log_str,
        filepath=config.log_path,
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

def stitch_patches(pred_label_folder, stitched_size=(1000,1000)) -> Tuple[List[np.array], str]:
    '''
    Stitch the patches together.

    '''
    imsize = 32 # TODO: add as param
    stitched_folder = pred_label_folder.replace('pred_patches', 'stitched_labels')
    colored_stitched_folder = pred_label_folder.replace('pred_patches', 'colored_stitched_labels')
    os.makedirs(stitched_folder, exist_ok=True)
    os.makedirs(colored_stitched_folder, exist_ok=True)

    print('pred_label_folder: ', pred_label_folder, ' stitched_folder: ', stitched_folder)

    label_list = sorted(glob(pred_label_folder + '/*.png'))

    base_label_list = []
    for label_path in label_list:
        # print('label_path: ', label_path)
        h, w = extract_h_w(label_path)
        h_w_string = f'_H{h}_W{w}'
        # print('label_path: ', label_path, h, w)
        h_w_string_idx = label_path.find(h_w_string)
        base_label_path = label_path[:h_w_string_idx] + '.png'

        if base_label_path not in base_label_list:
            # print('[True] base_label_path: ', base_label_path)
            base_label_list.append(base_label_path)


    stitched_label_list = []

    for base_label_path in base_label_list:
        # print('[Stitching] base_label_path: ', base_label_path)
        label_stitched = np.zeros((stitched_size[0], stitched_size[1]), dtype=np.uint8)
        label_patch_list = [item for item in label_list if base_label_path.replace('.png', '') in item]

        for label_patch_path in label_patch_list:
            h, w = extract_h_w(label_patch_path)

            offset_h = min(0, h) # negative in case of negative h or w
            offset_w = min(0, w)
            start_h = max(0, h)
            start_w = max(0, w)
            end_h = min(start_h + imsize + offset_h, stitched_size[0])
            end_w = min(start_w + imsize + offset_w, stitched_size[1])
            actual_height = end_h - start_h
            actual_width = end_w - start_w

            # print('label_patch h, w: ', h, w)
            # print('start_h, end_h, start_w, end_w: ', start_h, end_h, start_w, end_w)
            label_patch = cv2.imread(label_patch_path, cv2.IMREAD_GRAYSCALE)
            new_patch = label_patch[-offset_h:-offset_h + actual_height, -offset_w:-offset_w + actual_width]
            old_patch = label_stitched[start_h:end_h, start_w:end_w]
            # print('old_patch.shape: ', old_patch.shape, ' label_patch.shape: ', label_patch.shape, \
            #       ' new_patch.shape: ', new_patch.shape)

            updated_patch = np.maximum(old_patch, new_patch)
            #label_stitched[start_h:end_h, start_w:end_w] = updated_patch[:, :]
            label_stitched[start_h:end_h, start_w:end_w] = new_patch[:, :]

        stitched_label_list.append(label_stitched)

        save_path = base_label_path.replace(pred_label_folder, stitched_folder)
        cv2.imwrite(save_path, label_stitched)

        # save a colored version for visualization
        colored_label_stitched = np.zeros((label_stitched.shape[0], label_stitched.shape[1], 3), dtype=np.uint8)
        colored_label_stitched[label_stitched == 1] = (0, 255, 0)
        color_save_path = base_label_path.replace(pred_label_folder, colored_stitched_folder)
        cv2.imwrite(color_save_path, colored_label_stitched)

    log(f'Done stitching {len(label_list)} patches. Stitched: {len(stitched_label_list)}.')

    return stitched_label_list, stitched_folder

import utils.metrics as metrics
def eval_stitched(pred_folder, true_folder, organ='Colon', dataset_name='MoNuSeg') -> dict:
    '''
        Evaluation on final stitched label against the ground truth label.

    '''
    pred_list = sorted(glob(os.path.join(pred_folder + '/*.png')))
    true_list = sorted(glob(os.path.join(true_folder + '/*.png')))
    # Filter out other organs
    if dataset_name == 'MoNuSeg':
        from preprocessing.Metas import Organ2FileID
        file_ids = Organ2FileID[organ]['test']
    elif dataset_name == 'GLySAC':
        from preprocessing.Metas import GLySAC_Organ2FileID
        file_ids = GLySAC_Organ2FileID[organ]['test']
    true_list = [x for x in true_list if any([f'{file_id}' in x for file_id in file_ids])]
    print('pred_folder: ', pred_folder, '\ntrue_folder: ', true_folder)
    print(len(pred_list), len(true_list))

    assert len(pred_list) == len(true_list)

    metric_list = []
    for pred_label_path, true_label_path in zip(pred_list, true_list):
        pred_label = cv2.imread(pred_label_path, cv2.IMREAD_GRAYSCALE)
        true_label = cv2.imread(true_label_path, cv2.IMREAD_GRAYSCALE)
        assert pred_label.shape == true_label.shape

        metric = metrics.compute_metrics(pred_label, true_label, ['p_F1', 'aji', 'iou'])
        metric_list.append(metric)

    eval_results = {}
    for key in metric_list[0].keys():
        num = sum([i[key] for i in metric_list]) / len(metric_list)
        eval_results[key] = num

    return eval_results


import shutil

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

        test_label = warper(closest_label, flow=warp_field_forward)
    '''
    # NOTE: maybe we can even train on fly, for each pair.
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    _, _, _, test_set = prepare_dataset(config=config)

    pred_label_folder = os.path.join(config.output_save_root, model_name, 'pred_patches')
    # delete pred_label_folder
    if os.path.exists(pred_label_folder):
        shutil.rmtree(pred_label_folder)
        os.makedirs(pred_label_folder)
    else:
        os.makedirs(pred_label_folder)

    # Build the model
    warp_predictor = build_diffeomappingnet(config)
    warp_predictor.load_weights(config.DiffeoMappingNet_model_save_path, device=device)
    warp_predictor = warp_predictor.to(device)

    warper = Warper(size=config.target_dim)
    warper = warper.to(device)

    # Step 1: Load matched pairs.
    load_results = load_match_pairs(config.matched_pair_path_root, mode='infer', config=config)
    (test_images, test_labels, closest_images, closest_labels, test_image_paths) = load_results

    print('=====test_image_path====: ', test_image_paths[:10])
    if test_labels is not None:
        dataset = torch.utils.data.TensorDataset(closest_images, test_images, closest_labels, test_labels)
    else:
        dataset = torch.utils.data.TensorDataset(closest_images, test_images, closest_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Step 2: Predict & Apply the warping field.
    warp_predictor.eval()
    pred_label_list = []
    print(f'Starting inference for {test_images.shape[0]} test images ...')
    for iter_idx, batch in enumerate(tqdm(dataloader)):
        # [N, H, W] -> [N, 1, H, W]
        if test_labels is not None:
            (bclosest_images, btest_images, bclosest_labels, btest_labels) = batch
        else:
            (bclosest_images, btest_images, bclosest_labels) = batch

        if len(bclosest_labels.shape) == 3:
            bclosest_labels = bclosest_labels[:, None, ...]

        btest_images = btest_images.float().to(device)
        bclosest_images = bclosest_images.float().to(device)
        bclosest_labels = bclosest_labels.float().to(device)

        # Predict the warping field.
        warp_predicted = warp_predictor(torch.cat([bclosest_images, btest_images], dim=1))
        warp_field_forward = warp_predicted[:, :2, ...]
        warp_field_reverse = warp_predicted[:, 2:, ...]
        #print(warp_field_forward.shape, warp_field_reverse.shape)

        # Apply the warping field.
        images_U2A = warper(btest_images, flow=warp_field_forward)
        images_U2A_A2U = warper(images_U2A, flow=warp_field_reverse)
        pred_labels = warper(bclosest_labels, flow=warp_field_reverse)
        bpred_label_list = [m.cpu().detach().numpy() for m in pred_labels]
        pred_label_list.extend(bpred_label_list)

        plot_freq = 10
        shall_plot = iter_idx % plot_freq == 0
        #print('shall_plot: ', shall_plot, iter_idx, plot_freq, len(dataset))
        if shall_plot and test_labels is not None:
            save_path_fig_sbs = '%s/infer/figure_sample%s.png' % (
                save_folder, str(iter_idx).zfill(5))

            plot_side_by_side(save_path_fig_sbs, *numpy_variables(
                btest_images[0], bclosest_images[0],
                images_U2A_A2U[0], images_U2A[0],
                btest_labels[0] > 0.5, bclosest_labels[0] > 0.5,
                pred_labels[0] > 0.5))

    print('Completed inference.')

    print('Saving pred labels to disk ...')
    for i in range(len(pred_label_list)):
        # save to disk
        fname = os.path.join(pred_label_folder, os.path.basename(test_image_paths[i]))
        cv2.imwrite(fname, np.squeeze((pred_label_list[i] > 0.5).astype(np.uint8)))

    # Stitch the labels together.
    stitched_label_list, stitched_folder = stitch_patches(pred_label_folder)
    test_label_folder = config.groudtruth_folder
    stitched_results = eval_stitched(stitched_folder, test_label_folder, organ=config.organ, dataset_name=dataset_name)

    for k, v in stitched_results.items():
        log(F'[Eval] Stitched {k}: {v}', filepath=config.log_path, to_console=True)

    # if wandb_run is not None:
    #     for k, v in stitched_results.items():
    #         wandb_run.log({F'infer/stitched_{k}': v})

    if test_labels is not None:
        assert len(pred_label_list) == len(test_labels)
        print('len(test_labels), len(pred_label_list): ', len(test_labels), len(pred_label_list))

        # Step 3: Evaluation. Compute Dice Coeff.
        print(f'Computing Dice Coeff for {len(pred_label_list)} total labels...')
        dice_list = []
        iou_list = []

        # convert between torch Tensor & np array
        if 'torch' in str(type(test_labels)):
            test_labels = test_labels.cpu().detach().numpy()

        for i in range(len(pred_label_list)):
            # print('test_labels[i].shape, pred_label_list[i].shape: ', \
            #       test_labels[i].shape, pred_label_list[i].shape)
            dice_list.append(
                dice_coeff((np.expand_dims(test_labels[i], 0) > 0.5).transpose(1, 2, 0),
                        (pred_label_list[i] > 0.5).transpose(1, 2, 0)))
            iou_list.append(
                IoU((np.expand_dims(test_labels[i], 0) > 0.5).transpose(1, 2, 0),
                        (pred_label_list[i] > 0.5).transpose(1, 2, 0)))

        log('[Eval] Dice coeff (ours): %.3f \u00B1 %.3f.'
            % (np.mean(dice_list), np.std(dice_list)),
            filepath=config.log_path,
            to_console=True)
        log('[Eval] IoU (ours): %.3f \u00B1 %.3f.'
            % (np.mean(iou_list), np.std(iou_list)),
            filepath=config.log_path,
            to_console=True)

        if wandb_run is not None:
            wandb_run.log(
                {'infer/dice_seg_mean': np.mean(dice_list),
                 'infer/dice_seg_std': np.std(dice_list),
                 'infer/iou_seg_mean': np.mean(iou_list),
                 'infer/iou_seg_std': np.std(iou_list)})

    return


def main(config):
    assert config.mode in ['train', 'test', 'infer']

    # fix path issues
    ROOT = '/'.join(
        os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
    for key in vars(config).keys():
        if type(getattr(config, key)) == str and '$ROOT' in getattr(config, key):
            setattr(config, key, getattr(config, key).replace('$ROOT', ROOT))

    model_name = f'dataset-{config.dataset_name}_fewShot-{config.percentage:.1f}%_organ-{config.organ}'
    DiffeoMappingNet_str = f'DiffeoMappingNet_model-{config.DiffeoMappingNet_model}_hard-{config.hard_example_ratio}_epoch-{config.max_epochs}_smoothness-{config.coeff_smoothness}_seed{config.random_seed}'
    config.output_save_path = os.path.join(config.output_save_folder, model_name, DiffeoMappingNet_str, '')
    config.DiffeoMappingNet_model_save_path = os.path.join(config.output_save_path, 'model.ckpt')
    config.log_path = os.path.join(config.output_save_path, 'log.txt')

    # `config.n_views` set to 2 for DiffeoMappingNet training.
    config.n_views = 2

    seed_everything(config.random_seed)

    wandb_run = None
    if config.use_wandb and config.mode == 'train':
        wandb_run = wandb.init(
            entity=config.wandb_username,    # NOTE: need to use your wandb user name.
            project="cellseg",               # NOTE: need to create project on your wandb website.
            name=model_name + '_' + DiffeoMappingNet_str,
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

        train(config=config)
        test(config=config)
    elif config.mode == 'test':
        test(config=config)
    elif config.mode == 'infer':
        infer(config=config)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--mode', help='train|test|infer?', default='train')
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)
    parser.add_argument('--num-workers', help='Number of workers, e.g. use number of cores', default=4, type=int)

    parser.add_argument('--target-dim', default='(32, 32)', type=ast.literal_eval)
    parser.add_argument('--random-seed', default=1, type=int)

    parser.add_argument('--model-save-folder', default='$ROOT/checkpoints/', type=str)
    parser.add_argument('--output-save-folder', default='$ROOT/results/', type=str)

    parser.add_argument('--DiffeoMappingNet-model', default='VM-Diff', type=str)
    parser.add_argument('--dataset-name', default='MoNuSeg', type=str)
    parser.add_argument('--dataset-path', default='$ROOT/data/MoNuSeg/MoNuSegByCancer_patch_96x96/', type=str)
    parser.add_argument('--percentage', default=100, type=float)
    parser.add_argument('--organ', default=None, type=str)
    parser.add_argument('--depth', default=4, type=int)
    parser.add_argument('--latent-loss', default='SimCLR', type=str)
    parser.add_argument('--learning-rate', default=1e-3, type=float)
    parser.add_argument('--hard-example-ratio', default=0, type=float)
    parser.add_argument('--patience', default=100, type=int)
    parser.add_argument('--aug-methods', default='rotation,uniform_stretch,directional_stretch,volume_preserving_stretch,partial_stretch', type=str)
    parser.add_argument('--max-epochs', default=100, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--num-filters', default=32, type=int)
    parser.add_argument('--coeff-smoothness', default=0, type=float)
    parser.add_argument('--train-val-test-ratio', default='6:2:2', type=str)
    parser.add_argument('--n-plot-per-epoch', default=2, type=int)

    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-username', default='yale-cl2482', type=str)

    config = parser.parse_args()

    main(config)
