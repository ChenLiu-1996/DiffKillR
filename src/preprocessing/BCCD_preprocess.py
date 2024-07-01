'''
BCCD folder:
-/BCCD/JPEGImages
-/BCCD/Annotations
-/BCCD/ImageSets

Preprocess the BCCD dataset s.t. we have

/BCCD_patch_120x120/train/image
/BCCD_patch_120x120/test/image

/BCCD_patch_augmented_120x120/train/{aug_method}/image/{image_id}_{celltype}_H{startH}_W{startW}_{aug_id/original}.png


'''
import os
import sys
from glob import glob
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/augmentation/')
from aug_rotation import augment_rotation
from aug_stretch import augment_uniform_stretch, augment_directional_stretch, augment_volume_preserving_stretch
from aug_partial_stretch import augment_partial_stretch
from center_crop import center_crop


def patchify_image(img_path, annotation_path, patch_size, save_dir):
    '''
    Given an image and its annotation, patchify the image according to the annotation
    :param img_path: path to the image
    :param annotation_path: path to the annotation
    :param patch_size: size of the patch
    :param save_dir: directory to save the patches

    '''
    image_id = os.path.basename(img_path).split('.')[0] # 'BloodImage_00000'

    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert len(image.shape) == 3
    assert image.shape[-1] == 3

    tree = ET.parse(annotation_path)
    for obj in tree.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        label = obj.find('name').text

        centroid = ((xmin + xmax) // 2, (ymin + ymax) // 2)
        #print(f'Centroid: {centroid}')
        w_start = max(0, centroid[0] - patch_size // 2)
        w_end = min(image.shape[0], centroid[0] + patch_size // 2)
        h_start = max(0, centroid[1] - patch_size // 2)
        h_end = min(image.shape[1], centroid[1] + patch_size // 2)

        patch = image[w_start:w_end, h_start:h_end, :]

        # pad in case the patch exceeds the image size
        if patch.shape != (patch_size, patch_size, 3):
            #print(f'Patch shape: {patch.shape}')
            diff_w = patch_size - patch.shape[0]
            diff_h = patch_size - patch.shape[1]

            patch = np.pad(patch, ((0, diff_w), (0, diff_h), (0, 0)), mode='constant', constant_values=0)

        # save the patch
        name = f'{image_id}_{label}_H{h_start}_W{w_start}.png'
        save_path = os.path.join(save_dir, name)
        cv2.imwrite(save_path, patch)

    return


def augment_and_save(train_save_dir):
    '''
        train_save_dir: directory where all the train patches are saved

    '''
    # Now we augment the training patches
    aug_patch_size = 128
    resize_to = 32

    # randomly sample the patches
    percent = 0.1
    all_patches = glob(os.path.join(train_save_dir, '*.png'))
    train_patches = np.random.choice(all_patches, int(percent * len(all_patches)), replace=False)

    augmentation_methods = ['rotation',
                            'uniform_stretch',
                            'directional_stretch',
                            'volume_preserving_stretch',
                            'partial_stretch']
    multipliers = 1 # number of augmented patches per original patch
    cnt = 0
    celltype2cnt = {}
    for train_patch in train_patches:
        image = cv2.imread(train_patch, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert len(image.shape) == 3
        assert image.shape[-1] == 3
        #print('image:', image.shape, image)
        threshold = image.shape[0] * image.shape[1] * 3 * 255 * 0.2
        # print('threshold:', threshold)
        # print(np.sum(image))
        if np.sum(image) <= threshold: # skip empty patches
            #print('Skipping empty patch:', train_patch)
            continue

        #print('Not skipping:', train_patch)
        cnt += 1
        cell_type = os.path.basename(train_patch).split('_')[2] # 'RBC'
        if cell_type not in celltype2cnt:
            celltype2cnt[cell_type] = 1
        else:
            celltype2cnt[cell_type] += 1
        for aug_method in augmentation_methods:
            save_dir = f'../../data/{percent}_BCCD_patch_augmented_{resize_to}x{resize_to}/train/{aug_method}/image/'
            os.makedirs(save_dir, exist_ok=True)

            image_name = os.path.basename(train_patch).split('.')[0] # 'BloodImage_00000_RBC_H0_W0'
            # TODO: center crop & save original patch
            cropped_orig_img = center_crop(image=image, label=None, output_size=aug_patch_size)
            if resize_to != aug_patch_size:
                cropped_orig_img = cv2.resize(cropped_orig_img, (resize_to, resize_to), interpolation=cv2.INTER_CUBIC)
            save_path = os.path.join(save_dir, f'{image_name}_original.png')
            cv2.imwrite(save_path, cropped_orig_img)


            for i in range(multipliers):
                if aug_method == 'directional_stretch' or aug_method == 'volume_preserving_stretch':
                    image_aug, _ = globals()[f'augment_{aug_method}'](image, output_size=aug_patch_size)
                else:
                    image_aug = globals()[f'augment_{aug_method}'](image, output_size=aug_patch_size)

                aug_id = str(i).zfill(5)
                #print('aug_method:', aug_method, 'image_name:', image_name, 'aug_id:', aug_id)
                #print('image_aug:', image_aug.shape, 'image:', image.shape)
                assert len(image_aug.shape) == 3
                assert image_aug.shape[-1] == 3

                save_path = os.path.join(save_dir, f'{image_name}_aug{aug_id}.png')
                if resize_to != aug_patch_size:
                    image_aug = cv2.resize(image_aug, (resize_to, resize_to), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(save_path, image_aug)

    
    aug_cnt = cnt * multipliers * len(augmentation_methods)

    print(f'All train patches: {len(train_patches)}, qualified patches: {cnt}')
    print(f'Augmented {aug_cnt} patches, in total there are {cnt + aug_cnt} patches')
    print('All augmentations done!')

    # Check class type, balance, etc.
    print('Cell type distribution:')
    print(celltype2cnt)
    
    return


if __name__ == '__main__':
    # First prepare the train/test split and patchify the images.
    data_root = '../../external_data/BCCD'
    train_files = open(os.path.join(data_root, 'ImageSets/Main/trainval.txt'), 'r').readlines()
    train_files = [x.strip() for x in train_files]
    test_files = open(os.path.join(data_root, 'ImageSets/Main/test.txt'), 'r').readlines()
    test_files = [x.strip() for x in test_files]

    # Patchify the images according to the gt annotations
    img_dir = os.path.join(data_root, 'JPEGImages')
    annotation_dir = os.path.join(data_root, 'Annotations')

    patch_size = 240
    train_save_dir = f'../../data/BCCD_patch_{patch_size}x{patch_size}/train/image/'
    test_save_dir = f'../../data/BCCD_patch_{patch_size}x{patch_size}/test/image/'
    os.makedirs(train_save_dir, exist_ok=True)
    os.makedirs(test_save_dir, exist_ok=True)

    for img_id in tqdm(train_files):
        img_path = os.path.join(img_dir, img_id + '.jpg')
        annotation_path = os.path.join(annotation_dir, img_id + '.xml')
        patchify_image(img_path, annotation_path, patch_size, train_save_dir)
    print('Train patchification done!')

    for img_id in tqdm(test_files):
        img_path = os.path.join(img_dir, img_id + '.jpg')
        annotation_path = os.path.join(annotation_dir, img_id + '.xml')
        patchify_image(img_path, annotation_path, patch_size, test_save_dir)
    print('Test patchification done!')

    # Now we augment the training patches.
    augment_and_save(train_save_dir)

    




            

            
    

    


    





