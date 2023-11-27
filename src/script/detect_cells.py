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

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/')
from model.autoencoder import AutoEncoder
from utils.attribute_hashmap import AttributeHashmap
from utils.parse import parse_settings
from utils.prepare_dataset import prepare_dataset
from utils.seed import seed_everything



def find_cell_centroids(model: torch.nn.Module,
                        loader: torch.utils.data.DataLoader,
                        dataset: torch.utils.data.Dataset) -> Tuple[np.array]:
    '''
    Project the images in the dataset to the latent space via the model.
    Then, find the centroid of each cell type.
    '''
    latents = None
    cell_type_labels = []
    for _, _, canonical_images, _, image_paths in tqdm(loader):
        latent_, _ = model(canonical_images.float())
        if latents is None:
            latents = latent_.detach().cpu().numpy()
        else:
            latents = np.concatenate((latents, latent_.detach().cpu().numpy()), axis=0)

        for img_path in image_paths:
            cell_type = dataset.get_celltype(img_path=img_path)
            cell_type_labels.append(cell_type)

    latents = latents.reshape(latents.shape[0], -1)
    cell_type_labels = np.array(cell_type_labels)

    cell_type_centroids = {}
    for cell_type in np.unique(cell_type_labels):
        cell_type_centroids[cell_type] = latents[cell_type_labels == cell_type, ...].mean(axis=0)

    # from sklearn.metrics import pairwise_distances

    # pdist = pairwise_distances(latents[cell_type_labels == cell_type, ...])
    # pdist_arr = pdist[np.triu_indices(pdist.shape[0], k=1)]

    # import pdb
    # pdb.set_trace()

    return cell_type_centroids


def detect_cells(model: torch.nn.Module,
                 cell_type_centroids: Dict,
                 cell_type_map: Dict,
                 image: np.array,
                 patch_size: int = 96) -> np.array:
    '''
    Scan through the image with a sliding window,
    Project the image patch within the window to the latent space via the model.
    Classify the embedding into either one of the cell types or the background.
    '''

    dist_thr = 30.0
    cell_type_mask = np.zeros(image.shape[1:])

    for x_tl in tqdm(range(image.shape[1] - patch_size)):
        for y_tl in range(image.shape[2] - patch_size):
            if x_tl > 2:
                break
            patch = image[:, x_tl : x_tl + patch_size, y_tl : y_tl + patch_size]
            patch = torch.from_numpy(patch[None, ...]).float()
            curr_latent, _ = model(patch)
            curr_latent = curr_latent.detach().cpu().numpy()
            curr_latent = curr_latent.reshape(curr_latent.shape[0], -1)

            dist_nearest, cell_type_nearest = np.inf, None
            dist_map = {}
            for cell_type in cell_type_centroids.keys():
                dist_map[cell_type] = np.linalg.norm(curr_latent - cell_type_centroids[cell_type])
                if dist_map[cell_type] < dist_thr and dist_map[cell_type] < dist_nearest:
                    dist_nearest = dist_map[cell_type]
                    cell_type_nearest = cell_type

            if cell_type_nearest is not None:
                # NOTE: need to shift the values in `cell_type_map` up by 1.
                cell_type_mask[x_tl + patch_size//2,
                               y_tl + patch_size//2] = cell_type_map[cell_type_nearest] + 1

            # import pdb
            # pdb.set_trace()
    return cell_type_mask

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


def color_instances(label: np.array, palette: str = 'bright', n_colors: int = None) -> np.array:
    assert len(label.shape) == 2

    color_keys = np.unique(label)
    if color_keys[0] == 0:
        color_keys = color_keys[1:]
    else:
        warnings.warn('`color_instances`: color_keys[0] is not zero. It is %s instead.' % color_keys[0])

    if n_colors is None:
        n_colors = len(color_keys)
    color_values = sns.color_palette(palette, n_colors=n_colors)
    label_recolored = np.zeros((*label.shape, 3))

    for i, k in enumerate(color_keys):
        label_recolored[label==k, ...] = color_values[i]

    return label_recolored


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument('--num_workers', help='Number of workers, e.g. use number of cores', default=4)
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config.gpu_id = args.gpu_id
    config.num_workers = args.num_workers
    config.mode = 'test'
    config = parse_settings(config, log_settings=False)

    seed_everything(config.random_seed)

    image = load_image('../../data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00_patch_224x224/image/EndothelialCell_H7247_W11558_patch_224x224.png')
    try:
        model = globals()[config.model](num_filters=config.num_filters,
                                        in_channels=3,
                                        out_channels=3)
    except:
        raise ValueError('`config.model`: %s not supported.' % config.model)

    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model.load_weights(config.model_save_path, device=device)

    dataset, _, val_loader_tmp, _ = prepare_dataset(config=config)
    # val_loader.batch_size = 64
    val_loader = torch.utils.data.DataLoader(dataset=val_loader_tmp.dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers)
    cell_type_centroids = find_cell_centroids(model, val_loader, dataset)

    assert config.target_dim[0] == config.target_dim[1]
    cell_type_mask = detect_cells(model, cell_type_centroids, dataset.cell_type_to_idx, image, patch_size=config.target_dim[0])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(np.uint8(255.0 * (np.moveaxis(image, 0, -1) + 1) / 2))
    ax.set_axis_off()
    ax = fig.add_subplot(1, 2, 2)
    cell_type_mask_colored = np.uint8(255 * color_instances(label=cell_type_mask))
    mappable = ax.imshow(cell_type_mask, cmap='tab20b')

    # Colorbar.
    idx_to_cell_type = {0: 'Background'}
    for cell_type in dataset.cell_type_to_idx.keys():
        idx_to_cell_type[dataset.cell_type_to_idx[cell_type] + 1] = cell_type
    fmt = mpl.ticker.FuncFormatter(lambda x, pos: idx_to_cell_type[np.uint8(np.round(x))])
    ticks = list(idx_to_cell_type.keys())
    plt.colorbar(mappable, ax=ax, format=fmt, ticks=ticks, location='right', shrink=0.4)

    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig('detect_cells.png')

    import pdb
    pdb.set_trace()