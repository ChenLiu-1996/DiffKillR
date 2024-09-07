'''
Use DiffeoInvariantNet to detect and count cells in a given image.
Steps:
1. Load the image and label. Load the DiffeoInvariantNet model.
2. Convolve "model" with the image, assign the closest class in cell_bank to each patch.
3. Perform post-processing (NMS) to get rid of duplicate detections.

'''
import cv2
import os
import ast
import argparse
from glob import glob
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.prepare_dataset import prepare_dataset
from model.autoencoder import AutoEncoder
from datasets.MoNuSeg import MoNuSegDataset, load_image, load_label, normalize_image, fix_channel_dimension
from preprocessing.prepare_MoNuSeg import load_MoNuSeg_annotation
from blob_detector import detect_nuclei as blob_detect_nuclei

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model(model_path: str, num_filters: int, device: str = 'cpu') -> torch.nn.Module:
    """Load the DiffeoInvariantNet model."""
    model = globals()['AutoEncoder'](num_filters=num_filters,
                                    in_channels=3,
                                    out_channels=3)
    model = model.to(device)
    model.load_weights(model_path, device=device)
    print('%s: Model weights successfully loaded.' % model_path)
    model.eval()
    return model

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
                 cell_bank_patches: torch.Tensor, cell_bank_labels: np.array,
                 patch_size: int, stride: int, nms_threshold: float = 0.5,
                 device: str = 'cpu') -> List[Tuple[int, int, int, int, float]]:
    """
    Detect cells in the image using the DiffeoInvariantNet model.
    Args:
        image (np.array): The input image, shape (H, W, C).
        model (torch.nn.Module): The DiffeoInvariantNet model.
        cell_bank_patches (torch.Tensor): The cell bank patches, shape (N, C, patch_size, patch_size).
        cell_bank_labels (np.array): The cell bank labels, shape (N,). 0 for background, 1 for cell.
        patch_size (int): The size of the patches to extract.
        stride (int): The stride of the patches.
    Returns:
        List[Tuple[int, int, int, int, float]]: The detections, each detection is (min_x, min_y, max_x, max_y, score).
    """
    assert cell_bank_patches.shape[1] == image.shape[2]
    assert cell_bank_patches.shape[2] == cell_bank_patches.shape[3] == patch_size
    bs = 32

    # Obtain embeddings for the cell bank patches.
    print("Obtaining embeddings for the cell bank patches...")
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
    print("Extracting patches from the image...")
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
    print("Inferring embeddings for the patches...")
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

    print("Assigning labels to the patches...")
    # Assign the closest cell label to each patch.
    dists = torch.cdist(patch_embeddings, cell_bank_embeddings) # [M, N]
    closest_idx = torch.argmin(dists, dim=1) # [M]
    labels = cell_bank_labels[closest_idx]
    print(f"Cell percentage: {labels.mean()}")

    # Discard background patches.
    detections = []
    for i, label in enumerate(labels):
        if label == 1:
            min_x, min_y, max_x, max_y = coordinates[i]
            score = -dists[i, closest_idx[i]]
            detections.append((min_x, min_y, max_x, max_y, score))
    
    # Perform NMS
    print("Performing NMS...")
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

def evaluate_detections(detections: List[Tuple[int, int, int, int, float]], verts_list: List[np.array], image_sizes: Tuple[int, int]) -> Tuple[int, int, int]:
    """Evaluate the performance of the detections.
    Args:
        detections (List[Tuple[int, int, int, int, float]]): The detected bounding boxes (min_x, min_y, max_x, max_y, score).
        verts_list (List[np.array]): List of ground truth cell polygons. Each of (N, 2).
    Returns:
        Tuple[int, int, int]: True positives, false positives, and false negatives.
    """
    matched_gt = set()

    for detection in detections:
        min_x, min_y, max_x, max_y, _ = detection
        detection_box = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])

        for i, gt_cell in enumerate(verts_list):
            if i in matched_gt:
                continue

            # Check if the detection overlaps with the ground truth cell
            if inside_polygon(detection_box, gt_cell, image_sizes, threshold=1):
                matched_gt.add(i)
                break
    tp = len(matched_gt)
    fp = len(detections) - tp
    fn = len(verts_list) - len(matched_gt)

    return tp, fp, fn

def inside_polygon(detection_box, gt_cell, image_sizes, threshold=1):
    """Check if poly1 is inside poly2. Simply check if any pixel of poly1 is inside poly2.
    Args:
        detection_box (np.array): The first polygon, shape (N, 2).
        gt_cell (np.array): The second polygon, shape (M, 2).
        image_sizes (tuple): The size of the image.
        threshold (int): The minimum number of pixels for intersection.
    Returns:
        bool: True if poly1 is inside poly2, False otherwise.
    """
    # Check if any pixel of poly1 is inside poly2.
    mask1 = np.zeros(image_sizes, dtype=np.uint8)
    mask2 = np.zeros(image_sizes, dtype=np.uint8)

    poly1 = np.array(detection_box, dtype=np.int32).reshape((-1, 2))
    poly2 = np.array(gt_cell, dtype=np.int32).reshape((-1, 2))
    #print(poly1.shape, poly2.shape, mask1.shape, mask2.shape)

    cv2.fillPoly(mask1, [poly1], 1)
    cv2.fillPoly(mask2, [poly2], 1)
    intersection = np.logical_and(mask1, mask2)
    return intersection.sum() >= threshold

def visualize_detections(image_path: str, detections: List[Tuple[int, int, int, int, float]],
                         annotation_path: str=None, label_path: str=None):
    """Visualize the cell detections on the image."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    title_str = f'Detected: {len(detections)};'
    if annotation_path is not None:
        verts_list, region_id_list = load_MoNuSeg_annotation(annotation_path)
        title_str += f'GT: {len(verts_list)}'

        # Evaluate the performance.
        tp, fp, fn = evaluate_detections(detections, verts_list, image.shape[:2])
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        title_str += f'; Precision: {precision:.4f}; Recall: {recall:.4f}; F1: {f1:.4f}'
    if label_path is not None:
        label = np.array(cv2.imread(label_path, cv2.IMREAD_UNCHANGED)) # [H, W]
    
    # Blob detection
    blob_detections, im_with_keypoints = blob_detect_nuclei(image, return_overlay=True)
    print(f"[Blob Detection] Detected {len(blob_detections)} nuclei. Image Overlay: {im_with_keypoints.shape}, {image.shape}")
    blob_tp, blob_fp, blob_fn = evaluate_detections(blob_detections, verts_list, image.shape[:2])
    blob_precision = blob_tp / (blob_tp + blob_fp)
    blob_recall = blob_tp / (blob_tp + blob_fn)
    blob_f1 = 2 * blob_precision * blob_recall / (blob_precision + blob_recall)

    # Draw the detections on the image on the left image, and the label in the middle image, blob detector overlay on the right image.
    for (min_x, min_y, max_x, max_y, score) in detections:
        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    fig, axs = plt.subplots(1, 3, figsize=(5*3, 5))
    axs[0].imshow(image)
    axs[0].set_title(title_str)
    axs[0].axis('off')
    axs[1].imshow(label)
    axs[1].set_title('GT Mask')
    axs[1].axis('off')
    axs[2].imshow(im_with_keypoints)
    axs[2].set_title(f'Blob Detection: {len(blob_detections)}; Precision: {blob_precision:.4f}; Recall: {blob_recall:.4f}; F1: {blob_f1:.4f}')
    axs[2].axis('off')

    plt.show()

    # Save plot to file
    plt.savefig('detection_visualization.png', bbox_inches='tight', pad_inches=0)

def main(config):    
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = 'mps'

    model = load_model(config.model_path, num_filters=config.num_filters, device=device)
    image = load_image(config.image_path, target_dim=None)

    zero_pad = True # zero pad may help with counting cells near the image boundary
    if zero_pad:
        pad_size = (config.target_dim[0]//2, config.target_dim[1]//2)
        image = np.pad(image, ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (0, 0)), mode='constant')

    # Load the cell bank.
    dataset, train_loader, val_loader, test_loader = prepare_dataset(config=config)
    print('Cell bank loaded.')

    # Prepare cell bank patches and labels.
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    cell_bank_patches = []
    cell_bank_labels = []
    # print(dataset.patchID2Class)
    print('len of dataset:', len(dataset))
    for iter_idx, (_, _, _, _, canonical_images, _) in enumerate(dataloader):
        cell_bank_patches.append(canonical_images) # [B, C, H, W]
        for i in range(canonical_images.shape[0]):
            img_idx = iter_idx * 32 + i
            cell_patch_id = os.path.basename(dataset.img_paths[img_idx]).split('.')[0] # e.g. 'TCGA-A6-6782-01A-01-BS1_00000'
            cell_type = 1 if dataset.patchID2Class[cell_patch_id] in ['cell', 'Cell'] else 0
            cell_bank_labels.append(cell_type)

    cell_bank_patches = torch.cat(cell_bank_patches, dim=0) # [N, C, H, W]
    cell_bank_labels = np.array(cell_bank_labels) # [N]
    print(cell_bank_labels.shape, cell_bank_labels.shape)
    print(f"Cell bank loaded. {cell_bank_patches.shape[0]} patches, {cell_bank_labels.mean()} percentage of cells.")

    # Detect cells in the image.
    detections = detect_cells(image, model, cell_bank_patches, cell_bank_labels, 
                             config.patch_size, config.stride, config.nms_threshold)
    visualize_detections(config.image_path, detections, config.image_annotation_path, config.image_label_path)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cell Counting Entry point.')
    parser.add_argument("--image_path", type=str, default=ROOT_DIR + '/external_data/MoNuSeg/MoNuSegTestData/images/TCGA-AO-A0J2-01A-01-BSA.png')
    parser.add_argument("--model_path", type=str, default=ROOT_DIR + '/checkpoints/dataset-MoNuSeg_fewShot-100.0%_organ-Breast/DiffeoInvariantNet_model-AutoEncoder_depth-4_latentLoss-SimCLR_epoch-200_seed1.ckpt')
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument('--target-dim', default='(32, 32)', type=ast.literal_eval)

    parser.add_argument("--num_filters", type=int, default=32)
    parser.add_argument("--dataset_name", type=str, default='MoNuSeg')
    parser.add_argument('--dataset-path', default=ROOT_DIR + '/data/MoNuSeg2018TrainData_patch_96x96', type=str)
    parser.add_argument("--organ", type=str, default='Colon')
    parser.add_argument('--aug-methods', default='rotation,uniform_stretch,directional_stretch,volume_preserving_stretch,partial_stretch', type=str)
    parser.add_argument('--n-views', default=2, type=int)
    parser.add_argument('--train-val-test-ratio', default='6:2:2', type=str)
    parser.add_argument('--random_seed', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--nms_threshold", type=float, default=0.1)
    config = parser.parse_args()

    if config.organ is not None:
        config.dataset_path = f'{config.dataset_path}_{config.organ}'
    
    config.image_annotation_path = config.image_path.replace('/images/', '/').replace('.png', '.xml')
    config.image_label_path = config.image_path.replace('/images/', '/masks/')

    main(config)
