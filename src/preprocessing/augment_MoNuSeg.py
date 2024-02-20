'''
Read annotations from json file, find the label maps, and patchify around them.

'''
import cv2
import os
import numpy as np
from typing import List
from glob import glob
from tqdm import tqdm
import sys
import argparse

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
        os.makedirs(os.path.dirname(image_orig_path), exist_ok=True)
        os.makedirs(os.path.dirname(label_orig_path), exist_ok=True)
        cv2.imwrite(image_orig_path, cv2.cvtColor(image_orig, cv2.COLOR_RGB2BGR))
        cv2.imwrite(label_orig_path, label_orig)

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

            os.makedirs(os.path.dirname(image_aug_path), exist_ok=True)
            os.makedirs(os.path.dirname(label_aug_path), exist_ok=True)

            cv2.imwrite(image_aug_path, cv2.cvtColor(image_aug, cv2.COLOR_RGB2BGR))
            cv2.imwrite(label_aug_path, label_aug)

    return


def main():
    '''
    Main function.
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--patch_size', type=int, default=96)
    argparser.add_argument('--augmented_patch_size', type=int, default=32)
    argparser.add_argument('--percentage', type=float, default=0.01)
    argparser.add_argument('--multiplier', type=int, default=2)

    args = argparser.parse_args()
    patch_size = args.patch_size
    augmented_patch_size = args.augmented_patch_size
    percentage = args.percentage
    multiplier = args.multiplier

    patches_folder = '../../data/MoNuSeg2018TrainData_patch_%dx%d/' % (patch_size, 
                                                                       patch_size)
    image_path_list = sorted(glob(patches_folder + 'image/*.png'))
    label_path_list = sorted(glob(patches_folder + 'label/*.png'))

    # Subset of the data.
    #percentage = 0.01 # 1% of the data.
    total_cnt = int(len(image_path_list) * percentage)

    print('Total number of patches:', len(image_path_list),\
          'Percentage:', percentage, 'Total count:', total_cnt)

    augmented_folder = '../../data/%.3f_MoNuSeg2018TrainData_augmented_patch_%dx%d' % (
        percentage, augmented_patch_size, augmented_patch_size)

    prefix_list = []

    # Read the images and labels. Record the prefix and count the numbers for each cell type.
    # e.g. image path : 'TCGA-18-5592-01Z-00-DX1_H-1_W378_patch_96x96.png'
    # prefix: 'TCGA-18-5592-01Z-00-DX1_H-1_W378'
    for image_path, label_path in zip(image_path_list, label_path_list):
        prefix = os.path.basename(image_path).replace(
            '_patch_%sx%s.png' % (patch_size, patch_size), '')
        assert prefix == os.path.basename(label_path).replace(
            '_patch_%sx%s.png' % (patch_size, patch_size), '')
        prefix_list.append(prefix)

    # Augmentation.
    # Decide how many augmented versions per patch.
    multiplier = 2

    # Use a single data structure to hold all information for augmentation.
    augmentation_tuple_list = []

    prefix_list = prefix_list[:int(total_cnt)]
    image_path_list = image_path_list[:int(total_cnt)]
    label_path_list = label_path_list[:int(total_cnt)]

    for prefix, image_path, label_path in \
        zip(prefix_list, image_path_list, label_path_list):
        augmentation_tuple_list.append(
            (prefix, image_path, label_path, multiplier))
    augmentation_methods = ['rotation',
                            'uniform_stretch',
                            'directional_stretch',
                            'volume_preserving_stretch',
                            'partial_stretch']
    for augmentation_method in augmentation_methods:
        augment_and_save(augmentation_tuple_list, 
                         augmented_patch_size, 
                         augmented_folder, 
                         augmentation_method)

    print('Done.')
    print('Augmentation tuple list[:10]:')
    print(len(augmentation_tuple_list), augmentation_tuple_list[:10])
    print('Total number of patches:', \
          len(augmentation_tuple_list) * multiplier * len(augmentation_methods))


if __name__ == '__main__':
    main()