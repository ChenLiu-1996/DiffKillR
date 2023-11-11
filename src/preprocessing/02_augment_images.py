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

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/augmentation/')
from aug_rotation import augment_rotation
from aug_stretch import augment_uniform_stretch, augment_directional_stretch, augment_volume_preserving_stretch
from aug_partial_stretch import augment_partial_stretch
from center_crop import center_crop


class_value_map = {
    'EpithelialCell': 1,
    'EndothelialCell': 2,
    'Myocyte': 3,
    'Fibroblast': 4,
}


def load_and_get_flip_variants(image_path: str, label_path: str):
    '''
    Load the image and label from path,
    and return the 4 flip variants.
    '''
    image, label = load_image_and_label(image_path, label_path)
    image_label_flip_variants = get_flip_variants(image, label)
    return image_label_flip_variants


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


def get_flip_variants(image: np.array, label: np.array):
    '''
    Return the 4 flip variants of an image and its label.
    '''
    image_f1 = image
    label_f1 = label
    image_f2 = cv2.flip(image, 0)  # vertical flipping
    label_f2 = cv2.flip(label, 0)
    image_f3 = cv2.flip(image, 1)  # horizontal flipping
    label_f3 = cv2.flip(label, 1)
    image_f4 = cv2.flip(image, -1)  # vertical & horizontal flipping
    label_f4 = cv2.flip(label, -1)
    return [(image_f1, label_f1), (image_f2, label_f2),
            (image_f3, label_f3), (image_f4, label_f4)]



def augment_and_save(augmentation_tuple_list: List[tuple],
                     augmented_patch_size: int,
                     augmented_folder: str,
                     augmentation_method: str):
    '''
    Perform augmentation and save the augmented patches.
    '''
    for prefix, _, image_path, label_path, multiplier in tqdm(
            augmentation_tuple_list):

        image_label_flip_variants = load_and_get_flip_variants(
            image_path, label_path)

        # Save the unaugmented version.
        image_orig, label_orig = image_label_flip_variants[0]
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
        for image_, label_ in image_label_flip_variants:
            for _ in range(multiplier):
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
    patch_size = 224
    patches_folder = '../../data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00_patch_%dx%d/' % (
        patch_size, patch_size)
    image_path_list = sorted(glob(patches_folder + 'image/*.png'))
    label_path_list = sorted(glob(patches_folder + 'label/*.png'))

    augmented_patch_size = 96
    augmented_folder = '../../data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00_augmented_patch_%dx%d' % (
        augmented_patch_size, augmented_patch_size)

    prefix_list = []
    class_count_map = {}
    for cell_type in class_value_map.keys():
        class_count_map[cell_type] = 0

    # Read the images and labels. Record the prefix and count the numbers for each cell type.
    for image_path, label_path in zip(image_path_list, label_path_list):
        prefix = os.path.basename(image_path).replace(
            '_patch_%sx%s.png' % (patch_size, patch_size), '')
        assert prefix == os.path.basename(label_path).replace(
            '_patch_%sx%s.png' % (patch_size, patch_size), '')
        prefix_list.append(prefix)

        cell_type = prefix.split('_')[0]
        assert cell_type in class_value_map.keys()
        class_count_map[cell_type] += 1

    # Augmentation.
    # Decide how many augmented versions per cell in each cell type.
    augmentation_target_count = 1000  # In total, for each augmentation type.
    target_count_per_type = augmentation_target_count / len(
        class_value_map.keys())
    class_multiplier_map = {}
    for cell_type in class_value_map.keys():
        # 4 for 4 flipping variants.
        class_multiplier_map[cell_type] = max(1, round(target_count_per_type /
                                                       class_count_map[cell_type] / 4))

    # Use a single data structure to hold all information for augmentation.
    augmentation_tuple_list = []
    for prefix, image_path, label_path in \
        zip(prefix_list, image_path_list, label_path_list):
        cell_type = prefix.split('_')[0]
        multiplier = class_multiplier_map[cell_type]
        augmentation_tuple_list.append(
            (prefix, cell_type, image_path, label_path, multiplier))

    for augmentation_method in ['rotation',
                                'uniform_stretch',
                                'directional_stretch',
                                'volume_preserving_stretch',
                                'partial_stretch']:
        augment_and_save(augmentation_tuple_list, augmented_patch_size, augmented_folder, augmentation_method)


if __name__ == '__main__':
    main()