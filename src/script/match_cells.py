from typing import Dict, Tuple
import argparse
import numpy as np
import torch
import cv2
import os
import sys
import yaml
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
from sklearn.metrics import pairwise_distances
from glob import glob

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/')
from registration.spatial_transformer import SpatialTransformer as Warper
from model.autoencoder import AutoEncoder
from model.unet import UNet
from utils.attribute_hashmap import AttributeHashmap
from utils.parse import parse_settings
from utils.prepare_dataset import prepare_dataset
from utils.seed import seed_everything
from datasets.augmented import load_image


def load_image(path: str, target_dim: Tuple[int] = None) -> np.array:
    ''' Load image as numpy array from a path string.'''
    if target_dim is not None:
        image = np.array(
            cv2.resize(
                cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR),
                             code=cv2.COLOR_BGR2RGB), target_dim))
    else:
        image = np.array(
            cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR),
                         code=cv2.COLOR_BGR2RGB))

    # Normalize image.
    image = (image / 255 * 2) - 1

    # Channel last to channel first to comply with Torch.
    image = np.moveaxis(image, -1, 0)

    return image


class PatchesDataset(torch.utils.data.Dataset):
    def __init__(self,
                 target_dim=(32, 32),
                 data_dir='../../data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00_patch_32x32/image/'):

        self.target_dim = target_dim
        self.image_paths = glob(data_dir + '*.png')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = load_image(path=image_path, target_dim=self.target_dim)
        return image, image_path

class PairedPatchesDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_paths_seen,
                 image_paths_unseen,
                 seen_image_indices,
                 target_dim=(32, 32)):

        self.target_dim = target_dim
        self.image_paths_seen = image_paths_seen
        self.image_paths_unseen = image_paths_unseen
        self.seen_image_indices = seen_image_indices

    def __len__(self):
        return len(self.image_paths_unseen)

    def __getitem__(self, idx):
        unseen_image_path = self.image_paths_unseen[idx]
        seen_image_path = self.image_paths_seen[self.seen_image_indices[idx]]
        seen_mask_path = seen_image_path.replace('image', 'label')

        unseen_image = load_image(path=unseen_image_path, target_dim=self.target_dim)
        seen_image = load_image(path=seen_image_path, target_dim=self.target_dim)
        seen_mask= np.array(cv2.imread(seen_mask_path, cv2.IMREAD_UNCHANGED))
        return unseen_image, seen_image, seen_mask

def organize_seen_embedding(model: torch.nn.Module,
                            loader: torch.utils.data.DataLoader) -> Tuple[np.array]:
    '''
    Project the images in the dataset to the latent space via the model.
    Then, find the centroid of each cell type.

    NOTE: currently a bad hack: get the unique latent/image pair at the end.
    A lot of redundant seen images here, since we are iterative over all augmentations.
    '''
    latent_arr = None
    seen_image_path_arr = []
    for _, _, seen_images, _, _, seen_image_paths in tqdm(loader):
        _, latent = model(seen_images.float())
        latent = torch.flatten(latent, start_dim=1)
        if latent_arr is None:
            latent_arr = latent.detach().cpu().numpy()
        else:
            latent_arr = np.concatenate((latent_arr, latent.detach().cpu().numpy()), axis=0)
        seen_image_path_arr.extend([item for item in seen_image_paths])
    seen_image_path_arr = np.array(seen_image_path_arr)

    _, unique_indices = np.unique(latent_arr, return_index=True, axis=0)
    return latent_arr[unique_indices, :], seen_image_path_arr[unique_indices, ...]

def find_nearest_seen_image(model: torch.nn.Module,
                            loader: torch.utils.data.DataLoader,
                            seen_embeddings: np.array) -> np.array:
    '''
    For each input image, find the nearest seen image.
    The "nearest" is judged by latent space distance.
    '''
    seen_indices = None
    unseen_image_path_arr = []

    for unseen_image, unseen_image_path in tqdm(loader):
        _, latent = model(unseen_image.float())
        latent = torch.flatten(latent, start_dim=1)
        latent = latent.detach().cpu().numpy()

        # Find pairwise distance with embeddings of seen images.
        pdist = pairwise_distances(latent, seen_embeddings)

        if seen_indices is None:
            seen_indices = np.argmin(pdist, axis=1)
        else:
            seen_indices = np.concatenate((seen_indices, np.argmin(pdist, axis=1)), axis=0)

        unseen_image_path_arr.extend([item for item in unseen_image_path])

    unseen_image_path_arr = np.array(unseen_image_path_arr)
    return seen_indices, unseen_image_path_arr

def infer_reg2seg(model_reg2seg: torch.nn.Module,
                  loader: torch.utils.data.DataLoader) -> Tuple[np.array]:
    '''
    Run the Reg2Seg model.
    '''

    for iter_idx, (unseen_image, seen_image, seen_mask) in enumerate(tqdm(loader)):
        unseen_image = unseen_image.float()
        seen_image = seen_image.float()
        seen_mask = (seen_mask > 0.5).float()

        if len(seen_mask.shape) == 3:
            seen_mask = seen_mask[:, None, ...]

        # Predict the warping field.
        warp_predicted = model_reg2seg(torch.cat([seen_image, unseen_image], dim=1))
        warp_field_forward = warp_predicted[:, :2, ...]
        warp_field_reverse = warp_predicted[:, 2:, ...]

        # Apply the warping field.
        images_U2A = warper(unseen_image, flow=warp_field_forward)
        images_U2A_A2U = warper(images_U2A, flow=warp_field_reverse)
        masks_A2U = warper(seen_mask, flow=warp_field_reverse)

        mask_A = (seen_mask > 0.5).long()
        mask_A2U = (masks_A2U > 0.5).long()

        save_path_fig_sbs = './reg2seg_results/figure_sample%s.png' % (str(iter_idx).zfill(5))
        plot_side_by_side(save_path_fig_sbs, *numpy_variables( unseen_image[0], seen_image[0], images_U2A_A2U[0], images_U2A[0], mask_A[0], mask_A2U[0]))

def plot_side_by_side(save_path, im_U, im_A, im_U2A_A2U, im_U2A, ma_A, ma_A2U):
    plt.rcParams['font.family'] = 'serif'
    fig_sbs = plt.figure(figsize=(16, 8))

    ax = fig_sbs.add_subplot(2, 3, 1)
    ax.imshow(np.clip((im_U + 1) / 2, 0, 1))
    ax.set_title('Unannotated Image (U)')
    ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, 3, 2)
    ax.imshow(np.clip((im_A + 1) / 2, 0, 1))
    ax.set_title('Annotated Image (A)')
    ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, 3, 4)
    ax.imshow(np.clip((im_U2A_A2U + 1) / 2, 0, 1))
    ax.set_title('Cycled Image (U->A->U)')
    ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, 3, 5)
    ax.imshow(np.clip((im_U2A + 1) / 2, 0, 1))
    ax.set_title('Warped Image (U->A)')
    ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, 3, 3)
    ax.imshow(np.clip(ma_A, 0, 1), cmap='gray')
    ax.set_title('Annotated Mask (A)')
    ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, 3, 6)
    ax.imshow(np.clip(ma_A2U, 0, 1), cmap='gray')
    ax.set_title('Projected Mask (A->U)')
    ax.set_axis_off()

    fig_sbs.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig_sbs.savefig(save_path)
    plt.close(fig=fig_sbs)

    return

def numpy_variables(*tensors: torch.Tensor) -> Tuple[np.array]:
    '''
    Some repetitive numpy casting of variables.
    '''
    return [_tensor.cpu().detach().numpy().transpose(1, 2, 0) for _tensor in tensors]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument('--run_count', help='Provide this during testing!', default=1)
    parser.add_argument('--num_workers', help='Number of workers, e.g. use number of cores', default=4)
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config.gpu_id = args.gpu_id
    config.num_workers = args.num_workers
    config.mode = 'test'
    config = parse_settings(config, log_settings=False, run_count=args.run_count)

    seed_everything(config.random_seed)

    try:
        model_matching = globals()[config.model](num_filters=config.num_filters,
                                                 in_channels=3,
                                                 out_channels=3)
    except:
        raise ValueError('`config.model`: %s not supported.' % config.model)

    warper = Warper(size=config.target_dim)
    model_reg2seg = UNet(num_filters=64,
                         in_channels=6,
                         out_channels=4)

    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model_matching.load_weights(config.model_save_path, device=device)
    warper = warper.to(device)
    model_reg2seg.load_weights('../../checkpoints/run_1/reg2seg_UNet_internal_seed1.pty', device=device)

    _, train_loader_seen, _, _ = prepare_dataset(config=config)
    loader_seen = torch.utils.data.DataLoader(dataset=train_loader_seen.dataset,
                                              batch_size=64,
                                              shuffle=False,
                                              num_workers=config.num_workers)
    embeddings_seen, image_paths_seen = organize_seen_embedding(model_matching, loader_seen)

    dataset_unseen = PatchesDataset()
    loader_unseen = torch.utils.data.DataLoader(dataset=dataset_unseen,
                                                batch_size=256,
                                                shuffle=False,
                                                num_workers=config.num_workers)
    seen_image_indices, image_paths_unseen = find_nearest_seen_image(model_matching, loader_unseen, embeddings_seen)

    dataset_paired = PairedPatchesDataset(image_paths_seen, image_paths_unseen, seen_image_indices)
    loader_paired = torch.utils.data.DataLoader(dataset=dataset_paired,
                                                batch_size=256,
                                                shuffle=False,
                                                num_workers=config.num_workers)
    infer_reg2seg(model_reg2seg, loader_paired)
