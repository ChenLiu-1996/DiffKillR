'''
For the test set of MoNuSeg,
Instead of using the ground truth localization of nuclei,
We use template matching from the given train patches to detect the nuclei.
'''

import cv2
import os
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets.augmented_MoNuSeg import AugmentedMoNuSegDataset


def match_and_detect(images: torch.Tensor, templates: torch.Tensor, pool: str = 'max', 
                     window_size: int = 32,
                     stride: int = 32,
                     area_threshold: float = 0.6) -> list:
    '''
    Match batch images with the train patches and detect the nuclei.
    Input:
        - images: a numpy array of shape (n, C, H, W)
        - templates: a numpy array of shape (N, C, patch_H, patch_W)
        - pool: pool func to apply on out channels of activation map (N, C, H, W) -> (N, H, W)
        - window_size: the size of the moving window
        - stride: the stride of the moving window
        - area_threshold: the threshold to determine if a patch contains a nucleus
    Return:
        - all_nuclei_list: a list of list of nuclei coordinates; all_nuclei_list[i] is the list of coordinates for the i-th image.
    '''

    H, W = images.shape[-2:]
    N, C, patch_H, patch_W = templates.shape

    # TODO: Normalize the images and the templates to [0, 1]
    
    # Convolve the image with the templates -> convolution result is (n, N, H, W).
    # each template will produce an activation map (H, W) for each image
    activation_map = torch.nn.functional.conv2d(images,
                                                templates,
                                                stride=(1, 1),
                                                padding=(patch_H//2, patch_W//2))
    print('Done convolving, activation_map shape:', activation_map.shape)
    
    #assert activation_map.shape[-2:] == (H, W)
    assert activation_map.shape[1] == N

    # Apply pooling to get a single activation map for each image; activation_map shape: (n, H, W)
    if pool == 'max':
        activation_map = torch.max(activation_map, dim=1)[0]
    elif pool == 'mean':
        activation_map = torch.mean(activation_map, dim=1)
    else:
        raise ValueError('Invalid pool function. Use "max" or "mean".')
    
    print('Done pooling, activation_map shape:', activation_map.shape)

    # Binarize the activation map.
    pad_H = window_size // 2 # edge padding
    pad_W = window_size // 2

    all_nucleus_list = []

    activation_map = activation_map.cpu().numpy()
    for i in range(activation_map.shape[0]):
        threshold = np.mean(activation_map[i])
        activation_map[i] = (activation_map[i] < threshold) * 1.0
    
        # Moving window to detect the nuclei.
        act_map= np.pad(activation_map[i], ((pad_H, pad_H), (pad_W, pad_W)), mode='constant', constant_values=0)

        nucleus_list = []
        for h in range(pad_H, act_map.shape[0] - pad_H, stride):
            for w in range(pad_W, act_map.shape[1] - pad_W, stride):
                proposed_patch = act_map[h-window_size//2:h+window_size//2, w-window_size//2:w+window_size//2]
                # if h == pad_H or w == pad_W or h == act_map.shape[0] - pad_H or w == act_map.shape[1] - pad_W:
                #     # on the edge; only compute area that is not padded.
                #     proposed_patch = act_map[h:h+window_size//2, w:w+window_size//2]
                # else:
                #     proposed_patch = act_map[h-window_size//2:h+window_size//2, w-window_size//2:w+window_size//2]
                area = np.mean(proposed_patch)
                area_threshold = 0.6
                if area > area_threshold:
                    nucleus_list.append([h, w])
        
        all_nucleus_list.append(nucleus_list)
    
    return all_nucleus_list


def patchify_and_save(image, image_id, centroid_list,
                      patches_folder, patch_size):
    '''
    Divide the image into patches and save them.

    image: original image.
    image_id: id of the image. This should be unique.
    centroid_list: list of centroids for each polygon/cell.

    '''
    for centroid in tqdm(centroid_list, desc='patchifying...'):
        centroid[0], centroid[1] = int(centroid[0]), int(centroid[1])

        patch_image_path = '%s/image/%s_H%d_W%d_patch_%dx%d.png' % (
            patches_folder, image_id, centroid[0] - patch_size // 2,
            centroid[1] - patch_size // 2, patch_size, patch_size)
        os.makedirs(os.path.dirname(patch_image_path), exist_ok=True)

        h_begin = max(centroid[0] - patch_size // 2, 0)
        w_begin = max(centroid[1] - patch_size // 2, 0)
        h_end = min(h_begin + patch_size, image.shape[0])
        w_end = min(w_begin + patch_size, image.shape[1])

        patch_image = image[h_begin:h_end, w_begin:w_end, :]

        # Handle edge cases: literally on the edge.
        if patch_image.shape != (patch_size, patch_size, 3):
            h_diff = patch_size - patch_image.shape[0]
            w_diff = patch_size - patch_image.shape[1]
            patch_image = np.pad(patch_image,
                                 pad_width=((0, h_diff), (0, w_diff), (0, 0)),
                                 mode='constant')

        patch_image = cv2.cvtColor(patch_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(patch_image_path, patch_image)

    return


def process_MoNuSeg_Testdata(config, patch_size, device):
    folder = '../external_data/Chen_2024_MoNuSeg/MoNuSegTestData'

    annotation_files = sorted(glob(f'{folder}/*.xml'))
    image_files = sorted(glob(f'{folder}/*.tif'))

    all_centroids_list = []
    all_images = []
    for i, annotation_file in enumerate(tqdm(annotation_files)):
        image_id = os.path.basename(annotation_file).split('.')[0]
        image_file = f'{folder}/{image_id}.tif'
        if image_file not in image_files:
            print(f'Image file {image_file} not found.')
            continue

        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert len(image.shape) == 3
        assert image.shape[-1] == 3

        # image to (1, H, W, C)
        print('Image shape:', image.shape)

        all_images.append(image)
    all_images = np.stack(all_images, axis=0) # (n, H, W, C)

    # Load templates from train patches
    anchor_only = True # if True, only use the original images as templates
    aug_lists = config.aug_methods.split(',')
    dataset = AugmentedMoNuSegDataset(augmentation_methods=aug_lists,
                                         base_path=config.dataset_path,
                                         target_dim=config.target_dim)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.num_workers)
    
    anchor_bank = {'image': [], 'img_paths': [], 'sources': []}
    for iter_idx, (images, _, _, _, img_paths, cannoical_img_path) in enumerate(dataloader):
        images = images.float().to(device)

        for i in range(len(images)):
            if anchor_only:
                if 'original' in img_paths[i]:
                    anchor_bank['image'].append(images[i])
                    anchor_bank['img_paths'].append(img_paths[i])
                    anchor_bank['sources'].append('original')
            else:
                anchor_bank['image'].append(images[i])
                anchor_bank['img_paths'].append(img_paths[i])
                anchor_bank['sources'].append('original' if 'original' in img_paths[i] else 'augmented')
    
    anchor_bank['image'] = torch.stack(anchor_bank['image'], dim=0) # (N, C, H, W)

    templates = anchor_bank['image']

    # Match and detect the nuclei.
    all_images = torch.tensor(all_images).permute(0, 3, 1, 2).float().to(device)
    print(f'Matching {all_images.shape} images with {templates.shape} templates.')
    all_nuclei_list = match_and_detect(images=all_images,
                                      templates=templates.to(device),
                                      pool='max',
                                      window_size=patch_size,
                                      stride=patch_size)
    total = np.sum([len(nuclei_list) for nuclei_list in all_nuclei_list])
    print(f'Done matching and detecting nuclei. Total {total} detected. ')
    for i, annotation_file in enumerate(tqdm(annotation_files)):
        image_id = os.path.basename(annotation_file).split('.')[0]
        image_file = f'{folder}/{image_id}.tif'

        if image_file not in image_files:
            print(f'Image file {image_file} not found.')
            continue

        # Divide the image into patches.
        centroids_list = all_nuclei_list[i]
        patches_folder = '../data/MoNuSegTestData_TemplateLocalization_patch_%dx%d/' % (patch_size, patch_size)
        os.makedirs(patches_folder, exist_ok=True)
        patchify_and_save(image, image_id, centroids_list, patches_folder, patch_size)

    print('Done processing all images and annotations: annotated cells: %d' % total)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--aug_patch_size', type=int, default=32)
    args = parser.parse_args()

    model_config ='./config/MoNuSeg_AIAE.yaml'
    data_config ='./config/MoNuSeg_data.yaml'

    config = OmegaConf.merge(OmegaConf.load(model_config), OmegaConf.load(data_config))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    aug_patch_size = args.aug_patch_size
    process_MoNuSeg_Testdata(config, aug_patch_size, device)

