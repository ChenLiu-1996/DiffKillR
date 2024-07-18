from typing import Tuple, List
import argparse
import numpy as np
import torch
import cv2
import ast
import os
# from model.scheduler import LinearWarmupCosineAnnealingLR
from tqdm import tqdm
from matplotlib import pyplot as plt

from model.autoencoder import AutoEncoder
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

    fig_sbs.suptitle('IoU (label(U), label(A)) = %.3f, IoU (label(U), label(A->U)) = %.3f' % (
        IoU(ma_U, ma_A), IoU(ma_U, ma_A2U)), fontsize=15)
    fig_sbs.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig_sbs.savefig(save_path)
    plt.close(fig=fig_sbs)

    return

# @torch.no_grad()
# def generate_matching_pairs(config):
#     '''
#         Generate training pairs for DiffeoMappingNet model.
#         Given the trained DiffeoInvariantNet model, generate pairs of images from the training/val/test set.
#         Generate pairing, match augmented images to its non-mother anchor.
#         The anchor bank: training images from the original images.
#     '''

#     device = torch.device(
#         'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
#     dataset, train_set, val_set, test_set = prepare_dataset(config=config)

#     # Build the model
#     try:
#         model = globals()[config.DiffeoInvariantNet_model](num_filters=config.num_filters,
#                                                            in_channels=3,
#                                                            out_channels=3)
#     except:
#         raise ValueError('`config.DiffeoInvariantNet_model`: %s not supported.' % config.DiffeoInvariantNet_model)
#     model = model.to(device)

#     # Set all paths.
#     model.load_weights(config.DiffeoInvariantNet_model_save_path, device=device)
#     log('%s: Model weights successfully loaded.' % config.DiffeoInvariantNet_model,
#         to_console=True)
#     os.makedirs(config.output_save_path, exist_ok=True)

#     loader_map = {
#         'train': train_set,
#         'val': val_set,
#         'test': test_set,
#     }

#     import pdb; pdb.set_trace()
#     for k, loader in loader_map.items():

#         log(f'Generating {k} embeddings pairs ...')

#         # Generate embeddings.
#         to_be_matched_embeddings = []
#         to_be_matched_paths = []
#         bank_embeddings = []
#         bank_paths = []
#         pairing_save_path = os.path.join(config.output_save_path, f'{k}_pairs.csv')

#         for iter_idx, (images, _, image_n_view, canonical_images, _) in enumerate(loader):
#             images = images.float().to(device)  # [batch_size, C, H, W]

#             # [batch_size, n_views, C, H, W] -> [batch_size * n_views, C, H, W]
#             image_n_view = image_n_view.reshape(-1, image_n_view.shape[-3], image_n_view.shape[-2], image_n_view.shape[-1])
#             image_n_view = image_n_view.float().to(device)

#             _, latent_features = model(images)
#             latent_features = torch.flatten(latent_features, start_dim=1)
#             latent_features = latent_features.cpu().numpy()
#             for i in range(len(latent_features)):
#                 if 'original' in img_paths[i]:
#                     bank_embeddings.append(latent_features[i])
#                     bank_paths.append(img_paths[i])
#                 else:
#                     to_be_matched_embeddings.append(latent_features[i])
#                     to_be_matched_paths.append(img_paths[i])

#         to_be_matched_embeddings = np.stack(to_be_matched_embeddings, axis=0) # (N1, latent_dim)
#         bank_embeddings = np.stack(bank_embeddings, axis=0) # (N2, latent_dim)
#         log(f'[{k}]Bank: {bank_embeddings.shape}, \
#             To be matched: {to_be_matched_embeddings.shape}', to_console=True)

#         # Generate pairing, match augmented images to its non-mother anchor
#         dist_matrix = sklearn.metrics.pairwise_distances(to_be_matched_embeddings,
#                                             bank_embeddings,
#                                             metric='cosine') # [N1, N2]
#         closest_img_paths = []
#         closest_dists = []
#         for i in range(len(to_be_matched_embeddings)):
#             patch_id = dataset.get_patch_id(to_be_matched_paths[i])
#             mother = dataset.patch_id_to_canonical_pose_path[patch_id]
#             sorted_idx = np.argsort(dist_matrix[i])
#             # remove mother index from sorted_idx
#             if mother in bank_paths: # mother may not be in the same split
#                 mother_index = bank_paths.index(mother)
#                 sorted_idx = sorted_idx[sorted_idx != mother_index]

#             # select the closest non-mother index
#             closest_index = sorted_idx[0]
#             closest_img_paths.append(bank_paths[closest_index])
#             closest_dists.append(dist_matrix[i, closest_index]) # NOTE: this is wrong, but not affecting the result.

#         results_df = pd.DataFrame({
#             'test_image_path': to_be_matched_paths,
#             'closest_image_path': closest_img_paths,
#             'distance': closest_dists,
#             'source': ['original'] * len(to_be_matched_paths),
#         })
#         results_df.to_csv(pairing_save_path, index=False)
#         log(f'[{k}] Pairing saved to {pairing_save_path}', to_console=True)

#     return


# def load_match_pairs(matched_pair_path_root: str, mode: str, config):
#     '''
#     Load the matched pairs from csv. file and return
#         - unanotated_images
#         - unannotated_labels
#         - annotated_images
#         - annotated_labels
#         - unanonated_image_paths
#     '''
#     import pandas as pd
#     from datasets.augmented_MoNuSeg import load_image, load_label

#     matched_df = pd.read_csv(os.path.join(matched_pair_path_root, f'{mode}_pairs.csv'))
#     unanotated_image_paths = matched_df['test_image_path'].tolist()
#     annotated_image_paths = matched_df['closest_image_path'].tolist()
#     annotated_label_paths = [x.replace('image', 'label') for x in annotated_image_paths]
#     unannotated_label_paths = [x.replace('image', 'label') for x in unanotated_image_paths]

#     unannotated_images = [torch.Tensor(load_image(p, config.target_dim)) for p in unanotated_image_paths]
#     annotated_images = [torch.Tensor(load_image(p, config.target_dim)) for p in annotated_image_paths]
#     annotated_labels = [torch.Tensor(load_label(p, config.target_dim)) for p in annotated_label_paths]
#     if os.path.exists(unannotated_label_paths[0]):
#         unannotated_labels = [torch.Tensor(load_label(p, config.target_dim)) for p in unannotated_label_paths]
#     else:
#         unannotated_labels = None

#     unannotated_images = torch.stack(unannotated_images, dim=0) # (N, in_chan, H, W)
#     annotated_images = torch.stack(annotated_images, dim=0) # (N, in_chan, H, W)
#     annotated_labels = torch.stack(annotated_labels, dim=0) # (N, H, W)
#     if unannotated_labels is not None:
#         unannotated_labels = torch.stack(unannotated_labels, dim=0)

#     assert len(unannotated_images) == len(annotated_images) == len(annotated_labels)
#     print(f'Loaded {len(unannotated_images)} pairs of images and labels.')

#     return unannotated_images, unannotated_labels, \
#         annotated_images, annotated_labels, unanotated_image_paths

# FIXME!: I think we can even use another cycle loss: AM -> UM -> AM
# FIXME!: Also, if the augmented label is good, we can use it as a target for the forward cycle: UM -> AM.
def train(config, wandb_run=None):

    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')

    dataset, train_loader, val_loader, test_loader = prepare_dataset(config=config)

    # Build the model
    try:
        warp_predictor = globals()[config.DiffeoMappingNet_model](
            num_filters=config.num_filters,
            in_channels=6,
            out_channels=4)
    except:
        raise ValueError('`config.DiffeoMappingNet_model`: %s not supported.' % config.DiffeoMappingNet_model)

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
        plot_freq = int(len(train_loader) // config.n_plot_per_epoch)
        for iter_idx, (_, _, image_n_view, label_n_view, _, _) in enumerate(tqdm(train_loader)):

            assert image_n_view.shape[1] == 2
            curr_batch_size = image_n_view.shape[0]
            shall_plot = iter_idx % plot_freq == plot_freq - 1

            if not config.strong:
                # Weak mapping: map within the same cell (one augmented version to another).
                unannotated_images = image_n_view[:, 0, ...]
                unannotated_labels = label_n_view[:, 0, ...]
                annotated_images = image_n_view[:, 1, ...]
                annotated_labels = label_n_view[:, 1, ...]

            else:
                # Strong mapping: map across different cells.
                permuted_idx = np.random.permutation(len(image_n_view))
                unannotated_images = image_n_view[:, 0, ...]
                unannotated_labels = label_n_view[:, 0, ...]
                annotated_images = image_n_view[permuted_idx][:, 1, ...]
                annotated_labels = label_n_view[permuted_idx][:, 1, ...]


            if len(unannotated_labels.shape) == 3:
                unannotated_labels = unannotated_labels[:, None, ...]
            if len(annotated_labels.shape) == 3:
                annotated_labels = annotated_labels[:, None, ...]

            unannotated_images = unannotated_images.float().to(device) # (bsz, in_chan, H, W)
            annotated_images = annotated_images.float().to(device)
            unannotated_labels = unannotated_labels.float().to(device)
            annotated_labels = annotated_labels.float().to(device)

            if torch.is_floating_point(annotated_labels):
                annotated_labels = annotated_labels.float()
                unannotated_labels = unannotated_labels.float()
            else:
                # Only care about the binary label.
                assert annotated_labels.max() in [0, 1]
                annotated_labels = (annotated_labels > 0.5).float()
                unannotated_labels = (unannotated_labels > 0.5).float()

            # Predict the warping field.
            warp_predicted = warp_predictor(torch.cat([annotated_images, unannotated_images], dim=1))
            warp_field_forward = warp_predicted[:, :2, ...]
            warp_field_reverse = warp_predicted[:, 2:, ...]

            # Apply the warping field.
            images_U2A = warper(unannotated_images, flow=warp_field_forward)
            images_U2A_A2U = warper(images_U2A, flow=warp_field_reverse)
            labels_A2U = warper(annotated_labels, flow=warp_field_reverse)
            # print(labels_A2U.shape, unannotated_labels.shape, annotated_labels.shape)
            # print('check a few labels_A2U:', labels_A2U[0, ...], unannotated_labels[0, ...], annotated_labels[0, ...])

            # Compute Dice Coeff.
            for i in range(len(labels_A2U)):
                train_dice_ref_list.append(
                    dice_coeff((annotated_labels[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0),
                               (unannotated_labels[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0)))
                train_dice_seg_list.append(
                    dice_coeff((labels_A2U[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0),
                               (unannotated_labels[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0)))

            if shall_plot:
                save_folder = os.path.join(config.output_save_root, model_name, 'DiffeoMappingNet')
                save_path_fig_sbs = '%s/train/figure_log_epoch%s_sample%s.png' % (
                    save_folder, str(epoch_idx).zfill(5), str(iter_idx).zfill(5))
                plot_side_by_side(save_path_fig_sbs, *numpy_variables(
                    unannotated_images[0], annotated_images[0],
                    images_U2A_A2U[0], images_U2A[0],
                    unannotated_labels[0] > 0.5, annotated_labels[0] > 0.5,
                    labels_A2U[0] > 0.5))

            loss_forward = mse_loss(annotated_images, images_U2A)
            loss_cyclic = mse_loss(unannotated_images, images_U2A_A2U)

            loss = loss_forward + loss_cyclic
            train_loss += loss.item()
            train_loss_forward += loss_forward.item()
            train_loss_cyclic += loss_cyclic.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= train_loader.dataest
        train_loss_forward /= train_loader.dataest
        train_loss_cyclic /= train_loader.dataest

        lr_scheduler.step()

        log('Train [%s/%s] loss: %.3f, forward: %.3f, cyclic: %.3f, Dice coeff (ref): %.3f \u00B1 %.3f, Dice coeff (seg): %.3f \u00B1 %.3f.'
            % (epoch_idx + 1, config.max_epochs, train_loss, train_loss_forward, train_loss_cyclic,
               np.mean(train_dice_ref_list), np.std(train_dice_ref_list),
               np.mean(train_dice_seg_list), np.std(train_dice_seg_list)),
            filepath=config.log_dir,
            to_console=False)
        # if wandb_run is not None:
        #     wandb_run.log({'train/loss': train_loss,
        #                    'train/loss_forward': train_loss_forward,
        #                    'train/loss_cyclic': train_loss_cyclic,
        #                    'train/dice_ref_mean': np.mean(train_dice_ref_list),
        #                    'train/dice_ref_std': np.std(train_dice_ref_list),
        #                    'train/dice_seg_mean': np.mean(train_dice_seg_list),
        #                    'train/dice_seg_std': np.std(train_dice_seg_list)})

        # Validation.
        warp_predictor.eval()
        with torch.no_grad():
            val_loss, val_loss_forward, val_loss_cyclic = 0, 0, 0
            val_dice_ref_list, val_dice_seg_list = [], []
            plot_freq = int(len(val_test_set) // config.n_plot_per_epoch)
            # print('yo: ', plot_freq, len(val_test_loader))
            for iter_idx, (unannotated_images, unannotated_labels, annotated_images, annotated_labels) in enumerate(tqdm(val_test_loader)):
                shall_plot = iter_idx % plot_freq == plot_freq - 1

                # NOTE: batch size is len(val_set) here.
                # May need to change this if val_set is too large.

                if len(unannotated_labels.shape) == 3:
                    unannotated_labels = unannotated_labels[:, None, ...]
                if len(annotated_labels.shape) == 3:
                    annotated_labels = annotated_labels[:, None, ...]

                unannotated_images = unannotated_images.float().to(device) # (bsz, in_chan, H, W)
                annotated_images = annotated_images.float().to(device)
                unannotated_labels = unannotated_labels.float().to(device)
                annotated_labels = annotated_labels.float().to(device)
                # Only care about the binary label.
                annotated_labels = (annotated_labels > 0.5).float()
                unannotated_labels = (unannotated_labels > 0.5).float()

                # Predict the warping field.
                warp_predicted = warp_predictor(torch.cat([annotated_images, unannotated_images], dim=1))
                warp_field_forward = warp_predicted[:, :2, ...]
                warp_field_reverse = warp_predicted[:, 2:, ...]

                # Apply the warping field.
                images_U2A = warper(unannotated_images, flow=warp_field_forward)
                images_U2A_A2U = warper(images_U2A, flow=warp_field_reverse)
                labels_A2U = warper(annotated_labels, flow=warp_field_reverse)

                # Compute Dice Coeff.
                for i in range(len(labels_A2U)):
                    val_dice_ref_list.append(
                        dice_coeff((annotated_labels[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0),
                                   (unannotated_labels[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0)))
                    val_dice_seg_list.append(
                        dice_coeff((labels_A2U[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0),
                                   (unannotated_labels[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0)))

                if shall_plot:
                    save_folder = os.path.join(config.output_save_root, model_name, 'DiffeoMappingNet')
                    save_path_fig_sbs = '%s/val/figure_log_epoch%s_sample%s.png' % (
                        save_folder, str(epoch_idx).zfill(5), str(iter_idx).zfill(5))
                    plot_side_by_side(save_path_fig_sbs, *numpy_variables(
                        unannotated_images[0], annotated_images[0],
                        images_U2A_A2U[0], images_U2A[0],
                        unannotated_labels[0] > 0.5, annotated_labels[0] > 0.5,
                        labels_A2U[0] > 0.5))

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

        # if wandb_run is not None:
        #     wandb_run.log({'val/loss': val_loss,
        #                    'val/loss_forward': val_loss_forward,
        #                    'val/loss_cyclic': val_loss_cyclic,
        #                    'val/dice_ref_mean': np.mean(val_dice_ref_list),
        #                    'val/dice_ref_std': np.std(val_dice_ref_list),
        #                    'val/dice_seg_mean': np.mean(val_dice_seg_list),
        #                    'val/dice_seg_std': np.std(val_dice_seg_list)})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            warp_predictor.save_weights(config.DiffeoMappingNet_model_save_path)
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

    warp_predictor.load_weights(config.DiffeoMappingNet_model_save_path, device=device)
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

    for iter_idx, (unannotated_images, unannotated_labels, annotated_images, annotated_labels, _, _) in enumerate(tqdm(test_set)):
        shall_plot = iter_idx % plot_freq == plot_freq - 1

        if len(unannotated_labels.shape) == 3:
            unannotated_labels = unannotated_labels[:, None, ...]
        if len(annotated_labels.shape) == 3:
            annotated_labels = annotated_labels[:, None, ...]

        unannotated_images = unannotated_images.float().to(device) # (bsz, in_chan, H, W)
        annotated_images = annotated_images.float().to(device)
        unannotated_labels = unannotated_labels.float().to(device)
        annotated_labels = annotated_labels.float().to(device)
        # Only care about the binary label.
        annotated_labels = (annotated_labels > 0.5).float()
        unannotated_labels = (unannotated_labels > 0.5).float()

        # Predict the warping field.
        warp_predicted = warp_predictor(torch.cat([annotated_images, unannotated_images], dim=1))
        warp_field_forward = warp_predicted[:, :2, ...]
        warp_field_reverse = warp_predicted[:, 2:, ...]

        # Apply the warping field.
        images_U2A = warper(unannotated_images, flow=warp_field_forward)
        images_U2A_A2U = warper(images_U2A, flow=warp_field_reverse)
        labels_A2U = warper(annotated_labels, flow=warp_field_reverse)

        # Compute Dice Coeff.
        for i in range(len(labels_A2U)):
            test_dice_ref_list.append(
                dice_coeff((annotated_labels[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0),
                           (unannotated_labels[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0)))
            test_dice_seg_list.append(
                dice_coeff((labels_A2U[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0),
                           (unannotated_labels[i, ...] > 0.5).cpu().detach().numpy().transpose(1, 2, 0)))

        if shall_plot:
            save_path_fig_sbs = '%s/test/figure_log_sample%s.png' % (
                config.save_folder, str(iter_idx).zfill(5))
            plot_side_by_side(save_path_fig_sbs, *numpy_variables(
                unannotated_images[0], annotated_images[0],
                images_U2A_A2U[0], images_U2A[0],
                unannotated_labels[0] > 0.5, annotated_labels[0] > 0.5,
                labels_A2U[0] > 0.5))

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

    # Set all the paths.
    # dataset_name = config.dataset_name.split('_')[-1]
    # model_name = f'{config.percentage:.3f}_{config.organ}_m{config.multiplier}_{dataset_name}_depth{config.depth}_seed{config.random_seed}_{config.latent_loss}'
    # config.matched_pair_path_root = os.path.join(config.output_save_root, model_name)
    # config.model_save_path = os.path.join(config.output_save_root, model_name, 'DiffeoMappingNet.ckpt')
    # config.log_dir = os.path.join(config.log_folder, model_name) # This is log file path.
    # save_folder = os.path.join(config.output_save_root, model_name, 'DiffeoMappingNet')
    pred_label_folder = os.path.join(config.output_save_root, model_name, 'pred_patches')
    # delete pred_label_folder
    if os.path.exists(pred_label_folder):
        shutil.rmtree(pred_label_folder)
        os.makedirs(pred_label_folder)
    else:
        os.makedirs(pred_label_folder)

    # Build the model
    try:
        warp_predictor = globals()[config.model](num_filters=config.num_filters,
                                                 in_channels=6,
                                                 out_channels=4)
    except:
        raise ValueError('`config.model`: %s not supported.' % config.model)

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
        log(F'[Eval] Stitched {k}: {v}', filepath=config.log_dir, to_console=True)

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

        log('[Eval] Dice coeff (seg): %.3f \u00B1 %.3f.'
            % (np.mean(dice_list), np.std(dice_list)),
            filepath=config.log_dir,
            to_console=True)
        log('[Eval] IoU (seg): %.3f \u00B1 %.3f.'
            % (np.mean(iou_list), np.std(iou_list)),
            filepath=config.log_dir,
            to_console=True)

        # if wandb_run is not None:
        #     wandb_run.log({'infer/dice_seg_mean': np.mean(dice_list),
        #             'infer/dice_seg_std': np.std(dice_list),
        #             'infer/iou_seg_mean': np.mean(iou_list),
        #             'infer/iou_seg_std': np.std(iou_list)})

    return


from dotenv import load_dotenv

load_dotenv('../.env')
WANDB_ENTITY = os.getenv('WANDB_ENTITY')
PROJECT_PATH = os.getenv('PROJECT_PATH')

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
    parser.add_argument('--DiffeoMappingNet-model', default='UNet', type=str)
    parser.add_argument('--dataset-name', default='A28+axis', type=str)
    parser.add_argument('--percentage', default=100, type=float)
    parser.add_argument('--organ', default=None, type=str)
    parser.add_argument('--depth', default=4, type=int)
    parser.add_argument('--latent-loss', default='SimCLR', type=str)

    parser.add_argument('--learning-rate', default=1e-3, type=float)
    parser.add_argument('--strong', action='store_true', help='If true, we map among different cells.')
    parser.add_argument('--patience', default=50, type=int)
    parser.add_argument('--aug-methods', default='rotation,uniform_stretch,directional_stretch,volume_preserving_stretch,partial_stretch', type=str)
    parser.add_argument('--max-epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--num-filters', default=32, type=int)
    parser.add_argument('--train-val-test-ratio', default='6:2:2', type=str)
    parser.add_argument('--n-plot-per-epoch', default=2, type=int)

    config = parser.parse_args()
    assert config.mode in ['train', 'test', 'infer']

    # fix path issues
    ROOT = '/'.join(
        os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
    for key in vars(config).keys():
        if type(getattr(config, key)) == str and '$ROOT' in getattr(config, key):
            setattr(config, key, getattr(config, key).replace('$ROOT', ROOT))

    model_name = f'dataset-{config.dataset_name}_fewShot-{config.percentage:.1f}%_organ-{config.organ}_depth-{config.depth}_latentLoss-{config.latent_loss}_seed{config.random_seed}'
    config.DiffeoInvariantNet_model_save_path = os.path.join(config.model_save_folder, model_name, 'DiffeoInvariantNet.ckpt')
    config.DiffeoMappingNet_model_save_path = os.path.join(config.model_save_folder, model_name, 'DiffeoMappingNet.ckpt')

    config.output_save_path = os.path.join(config.output_save_folder, model_name, 'DiffeoMappingNet', '')
    config.log_path = os.path.join(config.output_save_folder, model_name, 'DiffeoMappingNet_log.txt')

    # `config.n_views` set to 2 for DiffeoMappingNet training.
    config.n_views = 2

    print(config)
    seed_everything(config.random_seed)

    # wandb_run = None
    # import wandb
    # if config.use_wandb and args.mode == 'train':
    #     wandb_run = wandb.init(
    #         entity=WANDB_ENTITY,
    #         project="cellseg",
    #         name=f"DiffeoMappingNet_{config.organ}_m{config.multiplier}_{config.dataset_name}_seed{config.random_seed}",
    #         config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    #         reinit=True,
    #         settings=wandb.Settings(start_method="thread")
    #     )

    if config.mode == 'train':
        # Initialize log file.
        log_str = 'Config: \n'
        for key in vars(config).keys():
            log_str += '%s: %s\n' % (key, getattr(config, key))
        log_str += '\nTraining History:'
        log(log_str, filepath=config.log_path, to_console=True)

        train(config=config)
        #test(config=config)
        infer(config=config)
    elif config.mode == 'test':
        pass
        #test(config=config)
    elif config.mode == 'infer':
        infer(config=config)

    # if wandb_run is not None:
    #     wandb_run.finish()