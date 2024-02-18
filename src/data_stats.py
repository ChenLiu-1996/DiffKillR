'''
This file contains the code to calculate the statistics of the data.

'''
import os
from glob import glob

if __name__ == '__main__':
    augmented_patch_size = 32
    augmented_folder = '../data/0.100_MoNuSeg2018TrainData_augmented_patch_%dx%d' % (
    augmented_patch_size, augmented_patch_size)
    augmentation_methods = ['rotation',
                                'uniform_stretch',
                                'directional_stretch',
                                'volume_preserving_stretch',
                                'partial_stretch']
    aug_image_paths = {}
    aug_label_paths = {}

    for augmentation_method in augmentation_methods:
        aug_image_path_list = sorted(glob(augmented_folder + '/' + augmentation_method + '/image/*.png'))
        aug_label_path_list = sorted(glob(augmented_folder + '/' + augmentation_method + '/label/*.png'))

        aug_image_paths[augmentation_method] = aug_image_path_list
        aug_label_paths[augmentation_method] = aug_label_path_list

        print('Augmentation method:', augmentation_method, 'Number of images:', len(aug_image_path_list))
        print('Augmentation method:', augmentation_method, 'Number of labels:', len(aug_label_path_list))


    total_image_paths = []
    total_label_paths = []

    for augmentation_method in augmentation_methods:
        total_image_paths += aug_image_paths[augmentation_method]
        total_label_paths += aug_label_paths[augmentation_method]

    print('Number of augmented images:', len(total_image_paths))
    print('Number of augmented labels:', len(total_label_paths))
    assert len(total_image_paths) == len(total_label_paths)

    print('Done.')


