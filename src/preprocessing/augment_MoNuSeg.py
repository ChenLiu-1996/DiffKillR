'''
Read the patches from the MoNuSeg2018TrainData_patch_96x96 folder.

Perform augmentation and save the augmented patches to the 
MoNuSeg2018TrainData_augmented_patch_32x32 folder, for each augmentation method.

'''

import cv2
import os
import numpy as np
from typing import List
from glob import glob
from tqdm import tqdm
import sys
import argparse
from omegaconf import OmegaConf

from Metas import Organ2FileID

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/augmentation/')
from aug_rotation import augment_rotation
from aug_stretch import augment_uniform_stretch, augment_directional_stretch, augment_volume_preserving_stretch
from aug_partial_stretch import augment_partial_stretch
from center_crop import center_crop


def load_image_and_label(image_path: str, label_path: str):
    '''
    Load the image and label from path.
    '''
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert len(image.shape) == 3
    assert image.shape[-1] == 3

    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    # assert len(label.shape) == 2

    patch_size_ = image.shape[0]
    assert patch_size_ == image.shape[1]
    assert patch_size_ == label.shape[0]
    assert patch_size_ == label.shape[1]

    return image, label


def augment_and_save(augmentation_tuple_list: List[tuple],
                     augmented_patch_size: int,
                     augmented_folder: str,
                     augmentation_method: str):
    '''
    Perform augmentation and save the augmented patches.
    '''
    for prefix, image_path, label_path, multiplier in tqdm(
            augmentation_tuple_list):

        # Save the unaugmented version.
        image_orig, label_orig = load_image_and_label(image_path, label_path)
        image_orig, label_orig = center_crop(image=image_orig,
                                             label=label_orig,
                                             output_size=augmented_patch_size)
        image_orig_path = '%s/%s/image/%s_original.png' % (
            augmented_folder, augmentation_method, prefix)
        label_orig_path = '%s/%s/label/%s_original.png' % (
            augmented_folder, augmentation_method, prefix)
        colored_label_orig_path = '%s/%s/colored_label/%s_original.png' % (
            augmented_folder, augmentation_method, prefix)
        os.makedirs(os.path.dirname(image_orig_path), exist_ok=True)
        os.makedirs(os.path.dirname(label_orig_path), exist_ok=True)
        os.makedirs(os.path.dirname(colored_label_orig_path), exist_ok=True)
        cv2.imwrite(image_orig_path, cv2.cvtColor(image_orig, cv2.COLOR_RGB2BGR))
        cv2.imwrite(label_orig_path, label_orig)
        colored_label_orig = np.zeros_like(image_orig)
        print(colored_label_orig.shape, label_orig.shape)
        colored_label_orig[label_orig == 1] = [0, 0, 255]
        cv2.imwrite(colored_label_orig_path, colored_label_orig)

        aug_counter = 0
        for _ in range(multiplier):
            image_, label_ = load_image_and_label(image_path, label_path)

            aug_counter += 1

            patch_size_ = image_.shape[0]
            assert patch_size_ > augmented_patch_size

            # Perform the augmentation.
            image_aug, label_aug = globals()['augment_' + augmentation_method](
                image=image_,
                label=label_,
                output_size=augmented_patch_size,
                random_seed=aug_counter,
            )[:2]

            assert image_aug.shape[0] == augmented_patch_size
            assert image_aug.shape[1] == augmented_patch_size
            assert label_aug.shape[0] == augmented_patch_size
            assert label_aug.shape[1] == augmented_patch_size

            image_aug_path = '%s/%s/image/%s_aug%s.png' % (
                augmented_folder, augmentation_method, prefix, str(aug_counter).zfill(5))
            label_aug_path = '%s/%s/label/%s_aug%s.png' % (
                augmented_folder, augmentation_method, prefix, str(aug_counter).zfill(5))
            colored_label_aug_path = '%s/%s/colored_label/%s_aug%s.png' % (
                augmented_folder, augmentation_method, prefix, str(aug_counter).zfill(5))

            os.makedirs(os.path.dirname(image_aug_path), exist_ok=True)
            os.makedirs(os.path.dirname(label_aug_path), exist_ok=True)
            os.makedirs(os.path.dirname(colored_label_aug_path), exist_ok=True)

            cv2.imwrite(image_aug_path, cv2.cvtColor(image_aug, cv2.COLOR_RGB2BGR))
            cv2.imwrite(label_aug_path, label_aug)
            colored_label_aug = np.zeros_like(image_aug)
            colored_label_aug[label_aug == 1] = [0, 0, 255]
            cv2.imwrite(colored_label_aug_path, colored_label_aug)


    return


def main():
    '''
    Main function.
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--patch_size', type=int, default=96)
    argparser.add_argument('--augmented_patch_size', type=int, default=32)
    argparser.add_argument('--percentage', type=float, default=0.1)
    argparser.add_argument('--multiplier', type=int, default=2)
    argparser.add_argument('--organ', type=str, default='Colon')

    args = argparser.parse_args()
    patch_size = args.patch_size
    augmented_patch_size = args.augmented_patch_size
    percentage = args.percentage
    multiplier = args.multiplier

    patches_folder = '../data/MoNuSeg2018TrainData_patch_%dx%d/' % (patch_size, 
                                                                       patch_size)
    image_path_list = sorted(glob(patches_folder + 'image/*.png'))
    label_path_list = sorted(glob(patches_folder + 'label/*.png'))

    # Subset of the data by organ.
    if args.organ in Organ2FileID:
        file_ids = Organ2FileID[args.organ]['train']
        image_path_list = [f for f in image_path_list if any(file_id in f for file_id in file_ids)]
        label_path_list = [f for f in label_path_list if any(file_id in f for file_id in file_ids)]
    else:
        print('Organ not found:', args.organ)
        return

    # Subset of the data by percentage.
    percentage = args.percentage
    total_cnt = int(len(image_path_list) * percentage)
    num_patches_per_image = total_cnt // len(file_ids)

    # uniform sampling among this organ images
    file_ids2cnt = {file_id: 0 for file_id in file_ids}
    subset_image_path_list = []
    for i in range(len(image_path_list)):
        file_id = os.path.basename(image_path_list[i]).split('_')[0]
        if file_ids2cnt[file_id] < num_patches_per_image:
            subset_image_path_list.append(image_path_list[i])
            file_ids2cnt[file_id] += 1
    
    if len(subset_image_path_list) < total_cnt:
        for image_path in image_path_list:
            if len(subset_image_path_list) >= total_cnt:
                break
            if image_path not in subset_image_path_list:
                subset_image_path_list.append(image_path)
    subset_label_path_list = [f.replace('image', 'label') for f in subset_image_path_list]

    print(f'All patches: {len(image_path_list)}, \
          percentage: {percentage}; \
            {args.organ} subset count: {total_cnt}, {len(subset_image_path_list)}')

    augmentation_methods = ['rotation',
                            'uniform_stretch',
                            'directional_stretch',
                            'volume_preserving_stretch',
                            'partial_stretch']

    augmented_folder = f'../data/{percentage:.3f}_{args.organ}_m{multiplier}_MoNuSeg2018TrainData_augmented_patch_{augmented_patch_size}x{augmented_patch_size}/'
    # Overwrite 'MoNuSeg_data.yaml' config file so models can find the data
    test_folder = f'../data/MoNuSeg2018TestData_patch_{augmented_patch_size}x{augmented_patch_size}/'
    conf = OmegaConf.create(
        {
            'dataset_name': 'MoNuSeg',
            'target_dim': [augmented_patch_size, augmented_patch_size],
            'dataset_path': augmented_folder,
            'aug_methods': ",".join(augmentation_methods),
            'test_folder': test_folder,
            'groudtruth_folder': '../external_data/Chen_2024_MoNuSeg/MoNuSegTestData/masks',
            'log_folder': '../logs/',
            'percentage': percentage,
            'organ': args.organ,
            'multiplier': multiplier,
        }
    )
    OmegaConf.save(conf, './config/MoNuSeg_data.yaml')

    if os.path.exists(augmented_folder):
        print(f'Augmented folder already exists at {augmented_folder}. \
                 Skipping sub-sampling & augmenting...')

        return

    prefix_list = []

    # Read the images and labels.
    # e.g. image path : 'TCGA-18-5592-01Z-00-DX1_H-1_W378_patch_96x96.png'
    # prefix: 'TCGA-18-5592-01Z-00-DX1_H-1_W378'
    for image_path, label_path in zip(subset_image_path_list, subset_label_path_list):
        prefix = os.path.basename(image_path).replace(
            '_patch_%sx%s.png' % (patch_size, patch_size), '')
        assert prefix == os.path.basename(label_path).replace(
            '_patch_%sx%s.png' % (patch_size, patch_size), '')
        prefix_list.append(prefix)

    '''Augmentation'''
    # Use a single data structure to hold all information for augmentation.
    augmentation_tuple_list = []
    for prefix, image_path, label_path in \
        zip(prefix_list, subset_image_path_list, subset_label_path_list):
        augmentation_tuple_list.append((prefix, image_path, label_path, multiplier))
    
    for augmentation_method in augmentation_methods:
        augment_and_save(augmentation_tuple_list, 
                         augmented_patch_size, 
                         augmented_folder, 
                         augmentation_method)

    print('Done.')
    print('Augmentation tuple list[:10]:')
    print(len(augmentation_tuple_list), augmentation_tuple_list[:10])
    print('Total number of patches:', \
          len(augmentation_tuple_list) * multiplier * len(augmentation_methods) \
            + len(augmentation_tuple_list))



if __name__ == '__main__':
    main()