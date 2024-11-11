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

def l1(im1, im2) -> float:
    '''
    Mean Absolute Error.
    '''
    return np.linalg.norm((im1.flatten() - im2.flatten()), ord=1)

def plot_side_by_side(save_path,
                      im_U, im_A,
                      im_U2A_A2U, im_U2A,
                      ma_U, ma_A, ma_A2U,
                      metric_name, metric):
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

    fig_sbs.suptitle('%s (label(U), label(A)) = %.3f, %s (label(U), label(A->U)) = %.3f' % (
        metric_name, metric(ma_U, ma_A), metric_name, metric(ma_U, ma_A2U)), fontsize=15)
    fig_sbs.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig_sbs.savefig(save_path)
    plt.close(fig=fig_sbs)

    return


def get_sample_image(image_path, image_shape = (256, 256)):
    from datasets.A28Axis import load_image, normalize_image

    image = normalize_image(load_image(image_path))
    full_image = image[:image_shape[0], :image_shape[1], ...]
    return full_image


def detect_nuclei(img: np.array, return_overlay: bool = False):
    if img.shape[-1] == 1:
        # (H, W, 1) to (H, W, 3)
        img = np.repeat(img, 3, axis=-1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Pad white the image to avoid border effects
    gray = cv2.copyMakeBorder(gray, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # params.minThreshold = 5
    # params.maxThreshold = 220

    # params.filterByArea = True
    # params.minArea = 150
    # params.maxArea = 10000.0

    # params.filterByCircularity = False
    # params.filterByConvexity = False
    # params.filterByInertia = False
    params.minConvexity = 0.8 #0.9499
    params.minDistBetweenBlobs = 1

    # # Create a detector with the parameters
    # detector = cv2.SimpleBlobDetector_create(params)
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(gray)

    nuclei_list = []
    for kp in keypoints:
        (w, h) = kp.pt
        nuclei_list.append([h, w])

    if return_overlay:
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return nuclei_list, im_with_keypoints
    else:
        return nuclei_list


def get_patches(full_image, nuclei_list, patch_size = (32, 32)):
    patch_list = []

    for centroid in nuclei_list:
        h_begin = max(0, int(np.round(centroid[0] - patch_size[0]/2)))
        h_end = min(full_image.shape[0], int(np.round(centroid[0] + patch_size[0]/2)))
        w_begin = max(0, int(np.round(centroid[1] - patch_size[1]/2)))
        w_end = min(full_image.shape[1], int(np.round(centroid[1] + patch_size[1]/2)))

        patch_image = full_image[h_begin:h_end, w_begin:w_end, :]
        if patch_image.shape != (patch_size, patch_size, 3):
            h_diff = patch_size[0] - patch_image.shape[0]
            w_diff = patch_size[1] - patch_image.shape[1]
            patch_image = np.pad(patch_image,
                                 pad_width=((0, h_diff), (0, w_diff), (0, 0)),
                                 mode='constant')

        patch_list.append(patch_image)

    return patch_list


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
        warp_predictor = globals()[config.DiffeoMappingNet_model](num_filters=config.num_filters,
                                                                  in_channels=6,
                                                                  out_channels=4)
    except:
        raise ValueError('`config.DiffeoMappingNet_model`: %s not supported.' % config.DiffeoMappingNet_model)

    latent_extractor.load_weights(config.DiffeoInvariantNet_model_save_path, device=device)
    latent_extractor = latent_extractor.to(device)

    warp_predictor.load_weights(config.DiffeoMappingNet_model_save_path, device=device)
    warp_predictor = warp_predictor.to(device)

    warper = Warper(size=config.target_dim)
    warper = warper.to(device)

    latent_extractor.eval()
    warp_predictor.eval()

    full_image = get_sample_image('../data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00_axis_patch_96x96/image/EpithelialCell_H3191_W6445_patch_96x96.png')
    nuclei_list, im_with_keypoints = detect_nuclei(np.uint8((full_image + 1) * 255 / 2), return_overlay=True)
    cv2.imwrite('example_blob_detection.png', im_with_keypoints)

    test_patches = get_patches(full_image, nuclei_list)

    _, train_loader, _, _ = prepare_dataset(config=config)
    bank_images, bank_labels, bank_embeddings = [], [], []
    for (_, _, _, _, canonical_images, canonical_labels) in train_loader:
        images = canonical_images.float().to(device)  # [batch_size, C, H, W]
        labels = canonical_labels.float().to(device)  # [batch_size, C, H, W]
        _, latent_features = latent_extractor(images)
        latent_features = torch.flatten(latent_features, start_dim=1)
        latent_features = latent_features.cpu().numpy()

        for batch_idx in range(len(images)):
            bank_images.append(images[batch_idx])
            bank_labels.append(labels[batch_idx])
            bank_embeddings.append(latent_features[batch_idx])

    stitched_label = np.zeros(full_image.shape[:2])
    for iter_idx, (patch_image_npy, centroid) in enumerate(tqdm(zip(test_patches, nuclei_list))):

        patch_image = torch.from_numpy(patch_image_npy.transpose(2, 0, 1)).unsqueeze(0).float()

        _, curr_embedding = latent_extractor(patch_image)
        curr_embedding = torch.flatten(curr_embedding, start_dim=1)
        curr_embedding = curr_embedding.cpu().numpy()

        dist_matrix = pairwise_distances(curr_embedding, bank_embeddings, metric='cosine')
        best_match_idx = np.argmin(dist_matrix)
        matched_bank_image = bank_images[best_match_idx].unsqueeze(0)
        matched_bank_label = bank_labels[best_match_idx].unsqueeze(0)

        label_is_binary = not torch.is_floating_point(matched_bank_label)

        if len(matched_bank_label.shape) == 3:
            matched_bank_label = matched_bank_label[:, None, ...]

        matched_bank_image = matched_bank_image.float().to(device) # (bsz, in_chan, H, W)
        matched_bank_label = matched_bank_label.float().to(device)

        if label_is_binary:
            matched_bank_label = (matched_bank_label > 0.5).float()

        # Predict the warping field.
        warp_predicted = warp_predictor(torch.cat([matched_bank_image, patch_image], dim=1))
        warp_field_forward = warp_predicted[:, :2, ...]
        warp_field_reverse = warp_predicted[:, 2:, ...]

        # Apply the warping field.
        images_U2A = warper(patch_image, flow=warp_field_forward)
        labels_A2U = warper(matched_bank_label, flow=warp_field_reverse)

        # Stitch label.
        patch_size = patch_image.shape[2:]
        h_begin = max(0, int(np.round(centroid[0] - patch_size[0]/2)))
        h_end = min(full_image.shape[0], int(np.round(centroid[0] + patch_size[0]/2)))
        w_begin = max(0, int(np.round(centroid[1] - patch_size[1]/2)))
        w_end = min(full_image.shape[1], int(np.round(centroid[1] + patch_size[1]/2)))

        curr_label = (labels_A2U.squeeze(0).squeeze(0) + 1) / 2
        if int(np.round(centroid[0] - patch_size[0]/2)) < 0:
            curr_label = curr_label[-int(np.round(centroid[0] - patch_size[0]/2)):, :]
        if int(np.round(centroid[0] + patch_size[0]/2)) > full_image.shape[0]:
            curr_label = curr_label[:full_image.shape[0] - int(np.round(centroid[0] + patch_size[0]/2)), :]
        if int(np.round(centroid[1] - patch_size[1]/2)) < 0:
            curr_label = curr_label[:, -int(np.round(centroid[1] - patch_size[1]/2)):]
        if int(np.round(centroid[1] + patch_size[1]/2)) > full_image.shape[1]:
            curr_label = curr_label[:, :full_image.shape[1] - int(np.round(centroid[1] + patch_size[1]/2))]

        stitched_label[h_begin:h_end, w_begin:w_end] = np.maximum(
            stitched_label[h_begin:h_end, w_begin:w_end], curr_label)

    cv2.imwrite('example_axis_prediction.png', np.uint8(stitched_label * 255))

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

from dotenv import load_dotenv

load_dotenv('../.env')
WANDB_ENTITY = os.getenv('WANDB_ENTITY')
PROJECT_PATH = os.getenv('PROJECT_PATH')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--mode', help='train|test|infer?', default='infer')
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

    if config.mode == 'infer':
        infer(config=config)

    # if wandb_run is not None:
    #     wandb_run.finish()