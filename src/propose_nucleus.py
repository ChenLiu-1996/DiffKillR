'''
This file is used to propose nuclei from the given image.
'''
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.augmented_MoNuSeg import AugmentedMoNuSegDataset, load_image

# from preprocessing import simple_nuclei_detector
import os
from glob import glob
from preprocessing.Metas import Organ2FileID
from matplotlib import pyplot as plt

from omegaconf import OmegaConf

def convolve_template(images: torch.Tensor, templates: torch.Tensor, pool: str = 'max') -> np.ndarray:
    """
    Convolve images with the given templates to produce an activation map.
    High values in the activation map indicate that the template is present in the image.

    Input:
    - images: a numpy array of shape (n, C, H, W)
    - templates: a numpy array of shape (N, C, patch_H, patch_W)
    - pool: pool func to apply on out channels of activation map (N, C, H, W) -> (N, H, W)
    Output:
    - activation_map: a numpy array of shape (n, H, W) # the same shape as the input image
    
    """
    H, W = images.shape[-2:]
    N, C, patch_H, patch_W = templates.shape

    # TODO: Normalize the images and the templates to [0, 1]
    
    # convolve the image with the templates -> convolution result is (n, N, H, W)
    # each template will produce an activation map (H, W) for each image
    activation_map = torch.nn.functional.conv2d(images,
                                                templates,
                                                stride=(1, 1),
                                                padding=(patch_H//2, patch_W//2))
    print('Done convolving, activation_map shape:', activation_map.shape)
    #assert activation_map.shape[-2:] == (H, W)
    assert activation_map.shape[1] == N

    # apply max pooling to the activation map to get a single activation map for each image
    if pool == 'max':
        activation_map = torch.max(activation_map, dim=1)[0]
    elif pool == 'mean':
        activation_map = torch.mean(activation_map, dim=1)
    else:
        raise ValueError('Invalid pool function. Use "max" or "mean".')
    
    
    return activation_map.cpu().numpy()

def detect_from_actmap(binary_activation_map: np.ndarray, threshold: float = 0.5, patch_size: int = 32) -> np.ndarray:
    """
    Detect the nuclei from the activation map using a threshold.
    Input:
        - binary_activation_map: a numpy array of shape (H, W) binarized
    Return:
        nucleus_list: a list of nuclei coordinates
    """
    # check patch by patch, stride , padding patch_H//2, patch_W//2
    # if the value is larger than the threshold, then it is a nucleus
    # the coordinates of the nucleus is the center of the patch
    
    # pad the activation map to avoid the edge
    pad_H = patch_size // 2
    pad_W = patch_size // 2
    binary_activation_map = np.pad(binary_activation_map, ((pad_H, pad_H), (pad_W, pad_W)), mode='constant', constant_values=0)

    nucleus_list = []
    stride = 32
    act_map = binary_activation_map
    for h in range(pad_H, act_map.shape[0] - pad_H, stride):
        for w in range(pad_W, act_map.shape[1] - pad_W, stride):
            if h == pad_H or w == pad_W or h == act_map.shape[0] - pad_H or w == act_map.shape[1] - pad_W:
                proposed_patch = act_map[h:h+patch_size//2, w:w+patch_size//2]
            else:
                proposed_patch = act_map[h-patch_size//2:h+patch_size//2, w-patch_size//2:w+patch_size//2]
            area = np.mean(proposed_patch)
            if area > threshold:
                nucleus_list.append([h, w])
    
    return nucleus_list

def overlay_nuclei(image: np.ndarray, nuclei_list: np.ndarray) -> np.ndarray:
    """
    Overlay the nuclei on the given image.
    Input:
        - image: a numpy array of shape (H, W, 3)
        - nuclei_list: a list of nuclei coordinates
    Return:
        - overlay: a numpy array of shape (H, W, 3)
    """
    overlay = image.copy()
    for h, w in nuclei_list:
        cv2.circle(overlay, (w, h), 5, (0, 255, 0), -1)
    return overlay


if __name__ == '__main__':
    model_config ='./config/MoNuSeg_AIAE.yaml'
    data_config ='./config/MoNuSeg_data.yaml'

    config = OmegaConf.merge(OmegaConf.load(model_config), OmegaConf.load(data_config))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    organ = 'Breast'
    print('Organ:', organ)
    file_ids = Organ2FileID[organ]['test']

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

    # Load test images
    img_path_list = [f'../external_data/Chen_2024_MoNuSeg/MoNuSegTestData/images/{file_id}.png' for file_id in file_ids]
    mask_path_list = [f'../external_data/Chen_2024_MoNuSeg/MoNuSegTestData/masks/{file_id}.png' for file_id in file_ids]
    image_list = [cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for image_path in img_path_list]
    mask_list = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for mask_path in mask_path_list]
    
    test_patch_folder = os.path.join('/gpfs/gibbs/pi/krishnaswamy_smita/dl2282/CellSeg/data/MoNuSeg2018TestData_patch_32x32', 'label')
    test_patch_files = sorted(glob(os.path.join(test_patch_folder, '*.png')))

    test_images = torch.tensor(image_list).permute(0, 3, 1, 2).float().to(device)
    print('Test images shape:', test_images.shape)

    activation_map = convolve_template(images=test_images, templates=templates, pool='max')
    print('Activation map shape:', activation_map.shape)
    print('Activation map max:', np.max(activation_map), np.min(activation_map), np.mean(activation_map), np.std(activation_map))

    # Visualize the activation map
    n = len(image_list)

    fig = plt.figure(figsize=(24, 6*n))
    for i in range(n):
        image = image_list[i]
        mask = mask_list[i]

        ax = fig.add_subplot(n, 4, 4*i+1)
        ax.imshow(image)
        ax.set_axis_off()

        # Detected nuclei
        ax = fig.add_subplot(n, 4, 4*i+2)
        ax.imshow(activation_map[i])
        ax.set_axis_off()

        # threshold the activation map to get the mask
        threshold = np.mean(activation_map)
        nuclei_mask = (activation_map[i] < threshold) * 1.0
        # ax = fig.add_subplot(n, 4, 4*i+3)
        # ax.imshow(nuclei_mask, cmap='gray')
        # ax.set_axis_off()

        # Propose nuclei
        nuclei_list = detect_from_actmap(nuclei_mask, threshold=0.6, patch_size=32)
        print(f'Proposed {len(nuclei_list)} nuclei.')

        # Overlay nuclei
        overlay = overlay_nuclei(image, nuclei_list)
        ax = fig.add_subplot(n, 4, 4*i+3)
        ax.imshow(overlay)
        ax.set_axis_off()
        ax.set_title(f'Proposed nuclei: {len(nuclei_list)}')
        
        # Groud truth mask
        filtered_patch_files = [x for x in test_patch_files if f'{file_ids[i]}' in x]
        print(f'Ground truth nuclei count: {len(filtered_patch_files)}')
        ax = fig.add_subplot(n, 4, 4*i+4)
        ax.imshow(mask, cmap='gray')
        ax.set_axis_off()
        ax.set_title(f'GT nuclei: {len(filtered_patch_files)}')

    
    fig.savefig('propose_nucleus.png')


