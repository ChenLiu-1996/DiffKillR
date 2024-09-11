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
from sklearn.metrics import pairwise_distances
from glob import glob
import skimage.measure
from skimage import morphology

from main_DiffeoMappingNet import build_diffeomappingnet, apply_flipping_rotation, predict_flipping_rotation

from model.autoencoder import AutoEncoder
from registration.spatial_transformer import SpatialTransformer as Warper
from registration.unet import UNet
from registration.voxelmorph import VxmDense as VoxelMorph
from registration.corrmlp import CorrMLP

from utils.attribute_hashmap import AttributeHashmap
from utils.prepare_dataset import prepare_dataset
from utils.log_util import log
from utils.seed import seed_everything

from datasets.MoNuSeg import normalize_image, fix_channel_dimension, load_image, load_label


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

def get_sample_image(image_path, image_shape = (256, 256)):
    image = normalize_image(load_image(image_path))
    if image_shape:
        image = image[:image_shape[0], :image_shape[1], ...]
    return image

def get_sample_label(label_path, image_shape = (256, 256)):
    label = load_label(label_path)
    if label.max() == 255:
        label = label / 255.0
    if label.shape[-1] == 3:
        assert (label[..., 0] == label[..., 1]).all()
        assert (label[..., 0] == label[..., 2]).all()
        label = label[..., 0]
    if image_shape:
        label = label[:image_shape[0], :image_shape[1], ...]
    return label


def get_patches(image, nuclei_list, patch_size = (32, 32)):
    patch_list = []

    for centroid in nuclei_list:
        h_begin = max(0, int(np.round(centroid[0] - patch_size[0]/2)))
        h_end = min(image.shape[0], int(np.round(centroid[0] + patch_size[0]/2)))
        w_begin = max(0, int(np.round(centroid[1] - patch_size[1]/2)))
        w_end = min(image.shape[1], int(np.round(centroid[1] + patch_size[1]/2)))

        patch_image = image[h_begin:h_end, w_begin:w_end, :]
        if patch_image.shape != (patch_size, patch_size, 3):
            h_diff = patch_size[0] - patch_image.shape[0]
            w_diff = patch_size[1] - patch_image.shape[1]
            patch_image = np.pad(patch_image,
                                 pad_width=((0, h_diff), (0, w_diff), (0, 0)),
                                 mode='constant')

        patch_list.append(patch_image)

    return patch_list


def mask_to_cell_centroid(mask):
    cell_loc_list = []
    labeled = skimage.measure.label(mask)

    for cell_idx in np.unique(labeled)[1:]:
        cell_loc = np.mean(np.argwhere(labeled == cell_idx), axis=0)
        h, w = cell_loc
        h = min(h, mask.shape[0])
        w = min(w, mask.shape[1])
        h = max(h, 0)
        w = max(w, 0)
        cell_loc_list.append((int(h), int(w)))
    return cell_loc_list

def extract_patches(image: np.array, patch_size: int, stride: int) -> Tuple[List[np.array], List[Tuple[int, int, int, int]]]:
    """Extract patches from the image. image: (H, W, C).
    Args:
        image (np.array): The input image, shape (H, W, C).
        patch_size (int): The size of the patches to extract.
        stride (int): The stride of the patches.
    Returns:
        List[np.array]: The patches, each patch is (patch_size, patch_size, C).
        List[Tuple[int, int, int, int]]: The coordinates of the patches, each coordinate is (min_x, min_y, max_x, max_y).
    """
    patches = []
    coordinates = []
    for i in range(0, image.shape[0] - patch_size + 1, stride):
        for j in range(0, image.shape[1] - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
            coordinates.append((i, j, i+patch_size, j+patch_size))
    return patches, coordinates

def detect_cells(image: np.array, model: torch.nn.Module,
                 cell_bank_patches: torch.Tensor, cell_bank_is_foreground: np.array,
                 patch_size: int, stride: int, nms_threshold: float = 0.5, voting_k: int = 1,
                 device: str = 'cpu') -> List[Tuple[int, int, int, int, float]]:
    """
    Detect cells in the image using the DiffeoInvariantNet model.
    Args:
        image (np.array): The input image, shape (H, W, C).
        model (torch.nn.Module): The DiffeoInvariantNet model.
        cell_bank_patches (torch.Tensor): The cell bank patches, shape (N, C, patch_size, patch_size).
        cell_bank_is_foreground (np.array): The cell bank labels, shape (N,). 0 for background, 1 for cell.
        patch_size (int): The size of the patches to extract.
        stride (int): The stride of the patches.
    Returns:
        List[Tuple[int, int, int, int, float]]: The detections, each detection is (min_x, min_y, max_x, max_y, score).
    """

    assert cell_bank_patches.shape[1] == image.shape[2]
    assert cell_bank_patches.shape[2] == cell_bank_patches.shape[3] == patch_size
    bs = 128

    # Obtain embeddings for the cell bank patches.
    print("[Step] Obtaining embeddings for the cell bank patches...")
    model.to(device)
    cell_bank_embeddings = []
    for i in range(0, cell_bank_patches.shape[0], bs):
        batch_patches = cell_bank_patches[i:i+bs] # list of [C, H, W]
        batch_patches = batch_patches.float().to(device)
        # print(batch_patches.shape)
        with torch.no_grad():
            _, ref_embeddings = model(batch_patches)
        ref_embeddings = ref_embeddings.flatten(start_dim=1) #[B, latent_dim]
        cell_bank_embeddings.append(ref_embeddings)
    cell_bank_embeddings = torch.cat(cell_bank_embeddings, dim=0) # [N, latent_dim]

    # Extract patches from the image.
    print("[Step] Extracting patches from the image...")
    patches_list, coordinates = extract_patches(image, patch_size, stride)
    patches_array = np.stack(patches_list, axis=0)
    print('Extracted patches:', patches_array.shape)
    C = image.shape[-1]

    patches = []
    for i, patch in enumerate(patches_array):
        patch = normalize_image(patch)
        patch = fix_channel_dimension(patch)
        patch = torch.from_numpy(patch).float().unsqueeze(0)
        patches.append(patch)
    patches = torch.cat(patches, dim=0)

    print("[Step] Inferring embeddings for the patches...")
    patch_embeddings = []
    for i in tqdm(range(0, len(patches), bs)):
        batch_patches = patches[i:i+bs]
        # print(batch_patches.shape)
        assert batch_patches.shape[1:] == (C, patch_size, patch_size)
        with torch.no_grad():
            _, embeddings = model(batch_patches)
        embeddings = embeddings.flatten(start_dim=1) #[B, latent_dim]
        patch_embeddings.append(embeddings)
    patch_embeddings = torch.cat(patch_embeddings, dim=0) # [M, latent_dim]

    print("[Step] Assigning labels to the patches...")
    # Assign the closest cell label to each patch. TODO: use voting instead of closest.
    dists = torch.cdist(patch_embeddings, cell_bank_embeddings) # [M, N]
    if voting_k == 1:
        closest_idx = torch.argmin(dists, dim=1) # [M]
        labels = cell_bank_is_foreground[closest_idx]
    else:
        closest_idx = torch.topk(dists, k=voting_k, dim=1, largest=False)[1] # [M, k]
        labels = cell_bank_is_foreground[closest_idx] # [M, k]
        labels = labels.mode(dim=1)[0] # [M]

    # Calculate uncertainty based on the distance to the closest cell and background embeddings
    cell_dists = dists[:, cell_bank_is_foreground == 1]
    background_dists = dists[:,  cell_bank_is_foreground == 0]
    #import pdb; pdb.set_trace()
    closest_cell_dist, _ = torch.min(cell_dists, dim=1) # [M]
    closest_background_dist, _ = torch.min(background_dists, dim=1) # [M]
    # if closest_cell_dist is small, closest_background_dist is large, uncertainty is small
    # if closest_cell_dist is large, closest_background_dist is small, uncertainty is large
    # uncertainty = closest_cell_dist / (closest_cell_dist + closest_background_dist)
    probs = np.exp(-closest_cell_dist) / (np.exp(-closest_cell_dist) + np.exp(-closest_background_dist) + 1e-6)
    print(f"Cell percentage: {labels.mean()}")
    print(f"Probs: {probs.mean()}")

    # Discard background patches.
    detections = []
    for i, label in enumerate(labels):
        if label == 1:
            min_x, min_y, max_x, max_y = coordinates[i]
            score = -dists[i, closest_idx[i]]
            detections.append((min_x, min_y, max_x, max_y, score.item()))

    # Filter out detections with low probs.
    before_filtering = len(detections)
    for i, det in enumerate(detections):
        if probs[i] < 0.5:
            detections.pop(i)
    after_filtering = len(detections)
    print(f"[Step] Filtered out {before_filtering - after_filtering} detections with low cell probability.")

    # Filter out detections with low scores.
    before_filtering = len(detections)
    score_mean, score_std = torch.tensor(np.array(detections)[:, 4]).mean(), torch.tensor(np.array(detections)[:, 4]).std()
    print(f"Score mean: {score_mean}, score std: {score_std} before filtering out detections with low scores...")
    detections = [det for det in detections if det[4] > score_mean + 0 * score_std]
    after_filtering = len(detections)
    print(f"[Step] Filtered out {before_filtering - after_filtering} detections with low scores.")

    print("[Step] Performing NMS...")
    detections = non_max_suppression(detections, nms_threshold)
    print(f"Detected {len(detections)} cells.")

    return detections

def non_max_suppression(boxes: List[Tuple[int, int, int, int, float]], threshold: float) -> List[Tuple[int, int, int, int, float]]:
    """Perform non-maximum suppression on the detections. Discard boxes with high IoUs.
    Args:
        boxes (List[Tuple[int, int, int, int, float]]): The boxes, each box is (min_x, min_y, max_x, max_y, score).
        threshold (float): The threshold for non-maximum suppression.
    Returns:
        List[Tuple[int, int, int, int, float]]: The final boxes after NMS.
    """

    def _compute_iou(box1: Tuple[int, int, int, int, float], box2: Tuple[int, int, int, int, float]) -> float:
        """Compute the IoU between two boxes. Each box is (min_x, min_y, max_x, max_y, score)"""
        x1_min, y1_min, x1_max, y1_max = box1[:4]
        x2_min, y2_min, x2_max, y2_max = box2[:4]

        # Calculate the intersection area
        inter_width = min(x1_max, x2_max) - max(x1_min, x2_min)
        inter_height = min(y1_max, y2_max) - max(y1_min, y2_min)
        inter_area = max(0, inter_width) * max(0, inter_height)

        # Calculate the union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        # Compute IoU
        iou = inter_area / union_area

        return iou

    sorted_boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    final_boxes = []
    while len(sorted_boxes) > 0:
        best_box = sorted_boxes.pop(0)
        final_boxes.append(best_box)

        # Compute IoU between best_box and other boxes
        ious = [_compute_iou(best_box, box) for box in sorted_boxes]
        sorted_boxes = [box for box, iou in zip(sorted_boxes, ious) if iou < threshold] # discard boxes with high IoUs

    return final_boxes


@torch.no_grad()
def infer(config: AttributeHashmap, n_plot_per_epoch: int = None):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')

    _, _, _, test_set = prepare_dataset(config=config)

    # Build the model
    try:
        latent_extractor = globals()[config.DiffeoInvariantNet_model](num_filters=config.num_filters,
                                                                      depth=config.depth,
                                                                      in_channels=3,
                                                                      out_channels=3)
    except:
        raise ValueError('`config.DiffeoInvariantNet_model`: %s not supported.' % config.DiffeoInvariantNet_model)

    try:
        warp_predictor = build_diffeomappingnet(config)
    except:
        raise ValueError('`config.DiffeoMappingNet_model`: %s not supported.' % config.DiffeoMappingNet_model)

    latent_extractor.load_weights(config.DiffeoInvariantNet_model_save_path, device=device)
    latent_extractor.to(device)

    warp_predictor.load_weights(config.DiffeoMappingNet_model_save_path, device=device)
    warp_predictor = warp_predictor.to(device)

    warper = Warper(size=config.target_dim)
    warper.to(device)

    latent_extractor.eval()
    warp_predictor.eval()

    dataset_trainval, _, _, _ = prepare_dataset(config=config)
    loader = torch.utils.data.DataLoader(dataset_trainval, batch_size=1, shuffle=False)
    bank_is_foreground, bank_images, bank_fg_images, bank_fg_labels, bank_fg_embeddings = [], [], [], [], []
    for (_, _, _, _, images_canonical, labels_canonical, is_foreground) in loader:
        images = images_canonical.float().to(device)  # [B, C, H, W]
        labels = labels_canonical.to(device)  # [B, C, H, W]
        _, latent_features = latent_extractor(images)
        latent_features = torch.flatten(latent_features, start_dim=1)
        latent_features = latent_features.cpu().numpy()

        for batch_idx in range(len(images)):
            bank_is_foreground.append(1 if is_foreground[batch_idx].item() else 0)
            bank_images.append(images[batch_idx])

            if is_foreground[batch_idx].item():
                bank_fg_images.append(images[batch_idx])
                bank_fg_labels.append(labels[batch_idx])
                bank_fg_embeddings.append(latent_features[batch_idx])

    bank_is_foreground = np.array(bank_is_foreground)

    infer_image_path_list = sorted(glob(os.path.join(config.infer_dataset_path, config.organ, 'test', 'images', '*.png')))
    infer_mask_path_list = sorted(glob(os.path.join(config.infer_dataset_path, config.organ, 'test', 'masks', '*.png')))
    assert len(infer_image_path_list) == len(infer_mask_path_list)

    for infer_image_path, infer_mask_path in zip(infer_image_path_list, infer_mask_path_list):
        infer_image = get_sample_image(infer_image_path)
        infer_mask = get_sample_label(infer_mask_path)

        dataset_folder = config.infer_dataset_path.split('/')[-1]
        if len(dataset_folder) == 0:
            dataset_folder = config.infer_dataset_path.split('/')[-2]

        # Detect cells in the image.
        if config.use_gt_loc:
            save_path = f'../comparison/results/{dataset_folder}/{config.organ}/Ours_gt_loc_seed{config.random_seed}/{os.path.basename(infer_image_path)}'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cell_centroid_list = mask_to_cell_centroid(infer_mask)
        else:
            save_path = f'../comparison/results/{dataset_folder}/{config.organ}/Ours_seed{config.random_seed}/{os.path.basename(infer_image_path)}'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            detections = detect_cells(infer_image, latent_extractor, torch.cat([item[None, ...] for item in bank_images], dim=0), bank_is_foreground,
                                      config.target_dim[0], stride=1, nms_threshold=0.05, voting_k=1)
            cell_centroid_list = [((item[0] + item[2])//2, (item[1] + item[3])//2) for item in detections]

        infer_patches = get_patches(infer_image, cell_centroid_list)

        stitched_label = np.zeros(infer_image.shape[:2])
        stitched_support = 1e-2 * np.ones_like(stitched_label)
        for iter_idx, (patch_image_npy, centroid) in enumerate(tqdm(zip(infer_patches, cell_centroid_list), total=len(infer_patches))):

            patch_image = torch.from_numpy(patch_image_npy.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            latent_extractor.to(device)

            _, curr_embedding = latent_extractor(patch_image)
            curr_embedding = torch.flatten(curr_embedding, start_dim=1)
            curr_embedding = curr_embedding.cpu().numpy()

            dist_matrix = pairwise_distances(curr_embedding, bank_fg_embeddings, metric='cosine')
            best_match_idx = np.argmin(dist_matrix)
            matched_bank_image = bank_fg_images[best_match_idx].unsqueeze(0)
            matched_bank_label = bank_fg_labels[best_match_idx].unsqueeze(0)

            label_is_binary = not torch.is_floating_point(matched_bank_label)

            if len(matched_bank_label.shape) == 3:
                matched_bank_label = matched_bank_label[:, None, ...]

            matched_bank_image = matched_bank_image.float().to(device) # (bsz, in_chan, H, W)
            matched_bank_label = matched_bank_label.float().to(device)

            if label_is_binary:
                matched_bank_label = (matched_bank_label > 0.5).float()

            # Predict flipping and rotation, and apply correction.
            flip_rot_transform_forward, _ = predict_flipping_rotation(patch_image, matched_bank_image)
            matched_bank_image_fliprot = apply_flipping_rotation(flip_rot_transform_forward, matched_bank_image)
            matched_bank_label_fliprot = apply_flipping_rotation(flip_rot_transform_forward, matched_bank_label)

            # Predict the warping field.
            warp_field_forward, warp_field_reverse = warp_predictor(source=patch_image, target=matched_bank_image_fliprot)

            # Apply the warping field.
            labels_A2U = warper(matched_bank_label_fliprot, flow=warp_field_reverse).cpu().detach().numpy()

            # Stitch label.
            patch_size = patch_image.shape[2:]
            h_begin = max(0, int(np.round(centroid[0] - patch_size[0]/2)))
            h_end = min(infer_image.shape[0], int(np.round(centroid[0] + patch_size[0]/2)))
            w_begin = max(0, int(np.round(centroid[1] - patch_size[1]/2)))
            w_end = min(infer_image.shape[1], int(np.round(centroid[1] + patch_size[1]/2)))

            curr_label = labels_A2U.squeeze(0).squeeze(0)
            if int(np.round(centroid[0] - patch_size[0]/2)) < 0:
                curr_label = curr_label[-int(np.round(centroid[0] - patch_size[0]/2)):, :]
            if int(np.round(centroid[0] + patch_size[0]/2)) > infer_image.shape[0]:
                curr_label = curr_label[:infer_image.shape[0] - int(np.round(centroid[0] + patch_size[0]/2)), :]
            if int(np.round(centroid[1] - patch_size[1]/2)) < 0:
                curr_label = curr_label[:, -int(np.round(centroid[1] - patch_size[1]/2)):]
            if int(np.round(centroid[1] + patch_size[1]/2)) > infer_image.shape[1]:
                curr_label = curr_label[:, :infer_image.shape[1] - int(np.round(centroid[1] + patch_size[1]/2))]

            stitched_label[h_begin:h_end, w_begin:w_end] = np.add(
                stitched_label[h_begin:h_end, w_begin:w_end], curr_label)
            stitched_support[h_begin:h_end, w_begin:w_end] += np.ones_like(curr_label)

        stitched_label = stitched_label / stitched_support
        stitched_label = stitched_label > 0.8
        stitched_label = morphology.remove_small_objects(stitched_label, 200)
        stitched_label = morphology.remove_small_holes(stitched_label)
        cv2.imwrite(save_path, np.uint8(np.uint8(stitched_label) * 255))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--mode', help='train|test|infer?', default='infer')
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)
    parser.add_argument('--num-workers', help='Number of workers, e.g. use number of cores', default=4, type=int)

    parser.add_argument('--target-dim', default='(32, 32)', type=ast.literal_eval)
    parser.add_argument('--random-seed', default=1, type=int)

    parser.add_argument('--dataset-name', default='MoNuSeg', type=str)
    parser.add_argument('--dataset-path', default='$ROOT/data/MoNuSeg/MoNuSegByCancer_patch_96x96/', type=str)
    parser.add_argument('--infer-dataset-path', default='$ROOT/data/MoNuSeg/MoNuSegByCancer_200x200/', type=str)
    parser.add_argument('--model-save-folder', default='$ROOT/checkpoints/', type=str)
    parser.add_argument('--output-save-folder', default='$ROOT/results/', type=str)

    parser.add_argument('--DiffeoInvariantNet-model', default='AutoEncoder', type=str)
    parser.add_argument('--DiffeoMappingNet-model', default='VM-Diff', type=str)
    parser.add_argument('--DiffeoInvariantNet-max-epochs', default=50, type=int)
    parser.add_argument('--DiffeoMappingNet-max-epochs', default=100, type=int)
    parser.add_argument('--percentage', default=10, type=float)
    parser.add_argument('--organ', default='Breast', type=str)
    parser.add_argument('--depth', default=4, type=int)
    parser.add_argument('--latent-loss', default='SimCLR', type=str)
    parser.add_argument('--use-gt-loc', action='store_true')

    parser.add_argument('--learning-rate', default=1e-3, type=float)
    parser.add_argument('--hard-example-ratio', default=0, type=float)
    parser.add_argument('--patience', default=50, type=int)
    parser.add_argument('--aug-methods', default='rotation,uniform_stretch,directional_stretch,volume_preserving_stretch,partial_stretch', type=str)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--num-filters', default=32, type=int)
    parser.add_argument('--coeff-smoothness', default=0, type=float)
    parser.add_argument('--train-val-test-ratio', default='6:2:2', type=str)
    parser.add_argument('--n-plot-per-epoch', default=2, type=int)
    parser.add_argument('--n-views', default=2, type=int)

    config = parser.parse_args()
    assert config.mode in ['train', 'test', 'infer']

    # fix path issues
    ROOT = '/'.join(
        os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
    for key in vars(config).keys():
        if type(getattr(config, key)) == str and '$ROOT' in getattr(config, key):
            setattr(config, key, getattr(config, key).replace('$ROOT', ROOT))

    model_name = f'dataset-{config.dataset_name}_fewShot-{config.percentage:.1f}%_organ-{config.organ}'
    DiffeoInvariantNet_str = f'DiffeoInvariantNet_model-{config.DiffeoInvariantNet_model}_depth-{config.depth}_latentLoss-{config.latent_loss}_epoch-{config.DiffeoInvariantNet_max_epochs}_seed-{config.random_seed}'
    config.DiffeoInvariantNet_model_save_path = os.path.join(os.path.join(config.output_save_folder, model_name, DiffeoInvariantNet_str, ''), 'model.ckpt')
    DiffeoMappingNet_str = f'DiffeoMappingNet_model-{config.DiffeoMappingNet_model}_hard-{config.hard_example_ratio}_epoch-{config.DiffeoMappingNet_max_epochs}_smoothness-{config.coeff_smoothness}_seed{config.random_seed}'
    config.DiffeoMappingNet_model_save_path = os.path.join(os.path.join(config.output_save_folder, model_name, DiffeoMappingNet_str, ''), 'model.ckpt')

    print(config)
    seed_everything(config.random_seed)

    if config.mode == 'infer':
        infer(config=config)
