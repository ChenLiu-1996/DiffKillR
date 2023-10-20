'''
Read annotations from json file, find the label maps, and patchify around them.
'''
import cv2
import os
import numpy as np
from typing import List
from glob import glob
from tqdm import tqdm

class_value_map = {
    'EpithelialCell': 1,
    'EndothelialCell': 2,
    'Myocyte': 3,
    'Fibroblast': 4,
}


def load_and_get_flip_variants(image_path: str, label_path: str):
    image, label = load_image_and_label(image_path, label_path)
    image_label_flip_variants = get_flip_variants(image, label)
    return image_label_flip_variants


def load_image_and_label(image_path: str, label_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert len(image.shape) == 3
    assert image.shape[-1] == 3

    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    assert len(label.shape) == 2

    patch_size_ = image.shape[0]
    assert patch_size_ == image.shape[1]
    assert patch_size_ == label.shape[0]
    assert patch_size_ == label.shape[1]
    return image, label


def get_flip_variants(image: np.array, label: np.array):
    image_f1 = image
    label_f1 = label
    image_f2 = cv2.flip(image, 0)  # vertical flipping
    label_f2 = cv2.flip(label, 0)
    image_f3 = cv2.flip(image, 1)  # horizontal flipping
    label_f3 = cv2.flip(label, 1)
    image_f4 = cv2.flip(image, -1)  # vertical & horizontal flipping
    label_f4 = cv2.flip(label, -1)
    return [(image_f1, label_f1), (image_f2, label_f2), (image_f3, label_f3),
            (image_f4, label_f4)]


def augment_and_save_translation(augmentation_tuple_list: List[tuple],
                                 augmented_patch_size: int,
                                 augmented_folder: str):
    '''
    Perform augmentation: tranlation
    and save the augmented patches.
    '''
    for prefix, _, image_path, label_path, multiplier in tqdm(
            augmentation_tuple_list):

        image_label_flip_variants = load_and_get_flip_variants(
            image_path, label_path)

        aug_counter = 0
        for image_, label_ in image_label_flip_variants:
            for _ in range(multiplier):
                aug_counter += 1

                patch_size_ = image_.shape[0]
                assert patch_size_ > augmented_patch_size
                max_displacement = (patch_size_ - augmented_patch_size) // 2
                tx = np.random.uniform(-max_displacement, max_displacement)
                ty = np.random.uniform(-max_displacement, max_displacement)
                translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

                center_crop_hw_begin = (patch_size_ -
                                        augmented_patch_size) // 2
                center_crop_hw_end = center_crop_hw_begin + augmented_patch_size
                image_translated = cv2.warpAffine(
                    image_, translation_matrix,
                    (patch_size_,
                     patch_size_))[center_crop_hw_begin:center_crop_hw_end,
                                   center_crop_hw_begin:center_crop_hw_end, :]
                label_translated = cv2.warpAffine(
                    label_,
                    translation_matrix, (patch_size_, patch_size_),
                    flags=cv2.INTER_NEAREST)[
                        center_crop_hw_begin:center_crop_hw_end,
                        center_crop_hw_begin:center_crop_hw_end]

                assert image_translated.shape[0] == augmented_patch_size
                assert image_translated.shape[1] == augmented_patch_size
                assert label_translated.shape[0] == augmented_patch_size
                assert label_translated.shape[1] == augmented_patch_size

                aug_image_path = '%s/translation/image/%s_aug%s.png' % (
                    augmented_folder, prefix, str(aug_counter).zfill(5))
                aug_label_path = '%s/translation/label/%s_aug%s.png' % (
                    augmented_folder, prefix, str(aug_counter).zfill(5))

                os.makedirs(os.path.dirname(aug_image_path), exist_ok=True)
                os.makedirs(os.path.dirname(aug_label_path), exist_ok=True)

                image_translated = cv2.cvtColor(image_translated,
                                                cv2.COLOR_RGB2BGR)
                cv2.imwrite(aug_image_path, image_translated)
                cv2.imwrite(aug_label_path, label_translated)

    return


def augment_and_save_rotation(augmentation_tuple_list: List[tuple],
                              augmented_patch_size: int,
                              augmented_folder: str):
    '''
    Perform augmentation: rotation
    and save the augmented patches.
    '''
    for prefix, _, image_path, label_path, multiplier in tqdm(
            augmentation_tuple_list):

        image_label_flip_variants = load_and_get_flip_variants(
            image_path, label_path)

        aug_counter = 0
        for image_, label_ in image_label_flip_variants:
            for _ in range(multiplier):
                aug_counter += 1

                patch_size_ = image_.shape[0]
                assert patch_size_ > augmented_patch_size
                angle = np.random.uniform(-180, 180)
                rotation_matrix = cv2.getRotationMatrix2D(
                    (patch_size_ / 2, patch_size_ / 2), angle, 1)
                center_crop_hw_begin = (patch_size_ -
                                        augmented_patch_size) // 2
                center_crop_hw_end = center_crop_hw_begin + augmented_patch_size
                image_rotated = cv2.warpAffine(
                    image_, rotation_matrix,
                    (patch_size_,
                     patch_size_))[center_crop_hw_begin:center_crop_hw_end,
                                   center_crop_hw_begin:center_crop_hw_end, :]
                label_rotated = cv2.warpAffine(
                    label_,
                    rotation_matrix, (patch_size_, patch_size_),
                    flags=cv2.INTER_NEAREST)[
                        center_crop_hw_begin:center_crop_hw_end,
                        center_crop_hw_begin:center_crop_hw_end]

                assert image_rotated.shape[0] == augmented_patch_size
                assert image_rotated.shape[1] == augmented_patch_size
                assert label_rotated.shape[0] == augmented_patch_size
                assert label_rotated.shape[1] == augmented_patch_size

                aug_image_path = '%s/rotation/image/%s_aug%s.png' % (
                    augmented_folder, prefix, str(aug_counter).zfill(5))
                aug_label_path = '%s/rotation/label/%s_aug%s.png' % (
                    augmented_folder, prefix, str(aug_counter).zfill(5))

                os.makedirs(os.path.dirname(aug_image_path), exist_ok=True)
                os.makedirs(os.path.dirname(aug_label_path), exist_ok=True)

                image_rotated = cv2.cvtColor(image_rotated, cv2.COLOR_RGB2BGR)
                cv2.imwrite(aug_image_path, image_rotated)
                cv2.imwrite(aug_label_path, label_rotated)
    return


def augment_and_save_stretch(augmentation_tuple_list: List[tuple],
                             augmented_patch_size: int, augmented_folder: str):
    '''
    Perform augmentation: stretch
    and save the augmented patches.
    '''
    for prefix, _, image_path, label_path, multiplier in tqdm(
            augmentation_tuple_list):

        image_label_flip_variants = load_and_get_flip_variants(
            image_path, label_path)

        aug_counter = 0
        for image_, label_ in image_label_flip_variants:
            for _ in range(multiplier):
                aug_counter += 1

                patch_size_ = image_.shape[0]
                assert patch_size_ > augmented_patch_size
                stretch_factor_h = np.random.uniform(0.8, 1.5)
                stretch_factor_w = np.random.uniform(0.8, 1.5)

                center_crop_hw_begin = (patch_size_ -
                                        augmented_patch_size) // 2
                center_crop_hw_end = center_crop_hw_begin + augmented_patch_size
                image_stretched = cv2.resize(
                    image_, (int(stretch_factor_h * patch_size_),
                             int(stretch_factor_w * patch_size_)
                             ))[center_crop_hw_begin:center_crop_hw_end,
                                center_crop_hw_begin:center_crop_hw_end, :]
                label_stretched = cv2.resize(
                    label_, (int(stretch_factor_h * patch_size_),
                             int(stretch_factor_w * patch_size_)),
                    interpolation=cv2.INTER_NEAREST)[
                        center_crop_hw_begin:center_crop_hw_end,
                        center_crop_hw_begin:center_crop_hw_end]

                assert image_stretched.shape[0] == augmented_patch_size
                assert image_stretched.shape[1] == augmented_patch_size
                assert label_stretched.shape[0] == augmented_patch_size
                assert label_stretched.shape[1] == augmented_patch_size

                aug_image_path = '%s/stretch/image/%s_aug%s.png' % (
                    augmented_folder, prefix, str(aug_counter).zfill(5))
                aug_label_path = '%s/stretch/label/%s_aug%s.png' % (
                    augmented_folder, prefix, str(aug_counter).zfill(5))

                os.makedirs(os.path.dirname(aug_image_path), exist_ok=True)
                os.makedirs(os.path.dirname(aug_label_path), exist_ok=True)

                image_stretched = cv2.cvtColor(image_stretched,
                                               cv2.COLOR_RGB2BGR)
                cv2.imwrite(aug_image_path, image_stretched)
                cv2.imwrite(aug_label_path, label_stretched)
    return


# from matplotlib import pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(2, 2, 1)
# ax.imshow(image_)
# ax = fig.add_subplot(2, 2, 2)
# ax.imshow(label_ * 100)
# ax = fig.add_subplot(2, 2, 3)
# ax.imshow(image_t)
# ax = fig.add_subplot(2, 2, 4)
# ax.imshow(label_t * 100)
# fig.savefig('wtf')


def augment_and_save_diffeomorphism(augmentation_tuple_list: List[tuple],
                                    augmented_patch_size: int,
                                    augmented_folder: str):
    '''
    Perform augmentation: diffeomorphism
    and save the augmented patches.
    '''
    # for prefix, _, image_path, label_path, multiplier in tqdm(augmentation_tuple_list):

    #     image_label_flip_variants = load_and_get_flip_variants(
    #         image_path, label_path)

    #     aug_counter = 0
    #     for image_, label_ in image_label_flip_variants:
    #         for _ in range(multiplier):
    #             aug_counter += 1

    #             patch_size_ = image_.shape[0]
    #             assert patch_size_ > augmented_patch_size

    #             # TODO: Write diffeomorphism.

    #             center_crop_hw_begin = (patch_size_ -
    #                                     augmented_patch_size) // 2
    #             center_crop_hw_end = center_crop_hw_begin + augmented_patch_size

    #             # TODO: Write diffeomorphism.
    #             image_diffeomorphism = None
    #             label_diffeomorphism = None

    #             assert image_diffeomorphism.shape[0] == augmented_patch_size
    #             assert image_diffeomorphism.shape[1] == augmented_patch_size
    #             assert label_diffeomorphism.shape[0] == augmented_patch_size
    #             assert label_diffeomorphism.shape[1] == augmented_patch_size

    #             aug_image_path = '%s/diffeomorphism/image/%s_aug%s.png' % (
    #                 augmented_folder, prefix, str(aug_counter).zfill(5))
    #             aug_label_path = '%s/diffeomorphism/label/%s_aug%s.png' % (
    #                 augmented_folder, prefix, str(aug_counter).zfill(5))

    #             os.makedirs(os.path.dirname(aug_image_path), exist_ok=True)
    #             os.makedirs(os.path.dirname(aug_label_path), exist_ok=True)

    #             image_diffeomorphism = cv2.cvtColor(image_diffeomorphism,
    #                                                 cv2.COLOR_RGB2BGR)
    #             cv2.imwrite(aug_image_path, image_diffeomorphism)
    #             cv2.imwrite(aug_label_path, label_diffeomorphism)
    return


if __name__ == '__main__':
    np.random.seed(1)

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
    augmentation_target_count = 2000  # In total, for each augmentation type.
    target_count_per_type = augmentation_target_count / len(
        class_value_map.keys())
    class_multiplier_map = {}
    for cell_type in class_value_map.keys():
        # 4 for 4 flipping variants.
        class_multiplier_map[cell_type] = round(target_count_per_type /
                                                class_count_map[cell_type] / 4)

    # Use a single data structure to hold all information for augmentation.
    augmentation_tuple_list = []
    for prefix, image_path, label_path in zip(prefix_list, image_path_list,
                                              label_path_list):
        cell_type = prefix.split('_')[0]
        multiplier = class_multiplier_map[cell_type]
        augmentation_tuple_list.append(
            (prefix, cell_type, image_path, label_path, multiplier))

    augment_and_save_translation(augmentation_tuple_list, augmented_patch_size,
                                 augmented_folder)

    augment_and_save_rotation(augmentation_tuple_list, augmented_patch_size,
                              augmented_folder)

    augment_and_save_stretch(augmentation_tuple_list, augmented_patch_size,
                             augmented_folder)

    augment_and_save_diffeomorphism(augmentation_tuple_list,
                                    augmented_patch_size, augmented_folder)
