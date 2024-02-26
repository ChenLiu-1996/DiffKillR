import itertools
import os
from typing import Literal
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from time import time

class AugmentedMoNuSegDataset(Dataset):
    def __init__(self,
                 augmentation_methods: List[str],
                 base_path: str = '../data/0.010_MoNuSeg2018TrainData_augmented_patch_32x32',
                 target_dim: Tuple[int] = (32, 32)):

        super().__init__()

        self.target_dim = target_dim
        self.augmentation_methods = augmentation_methods
        self.augmentation_folders = [
            folder for folder in sorted(glob('%s/*/' % base_path))
            if folder.split('/')[-2] in augmentation_methods
        ]
        self.augmentation_method_to_folder = {}
        for folder in self.augmentation_folders:
            self.augmentation_method_to_folder[folder.split('/')[-2]] = folder
        
        # Note: make sure only 1 copy of patch_id_original.png exists in dataset
        # Otherwise, can't perform super contrastive learning.
        '''
            e.g. 'TCGA-B0-5710-01Z-00-DX1_H172_W505_original.png', 
                 'TCGA-NH-A8F7-01A-01-TS1_H673_W415_aug00001.png'
        '''            
        self.patch_id_to_canonical_pose_path = {}
        self.patch_id_to_patch_id_idx = {}

        self.all_image_paths = []
        self.all_label_paths = []

        for folder in self.augmentation_folders:
            aug_method = folder.split('/')[-2]
            #print('Loading images and labels from %s; aug_method: %s ' % (folder, aug_method))
            
            img_paths = sorted(glob('%s/image/*.png' % (folder)))
            label_paths = sorted(glob('%s/label/*.png' % (folder)))
            for img_path, label_path in zip(img_paths, label_paths):
                file_name = os.path.basename(img_path)
                img_path = os.path.join(base_path, aug_method, 'image', file_name)
                label_path = os.path.join(base_path, aug_method, 'label', file_name)
                #print(img_path)
                patch_id = "_".join(file_name.split('_')[:3]) # e.g. 'TCGA-B0-5710-01Z-00-DX1_H172_W505'
                is_original = file_name.split('_')[-1].split('.')[0] == 'original'
                
                if is_original == True:
                    if patch_id not in self.patch_id_to_canonical_pose_path.keys():
                        self.patch_id_to_canonical_pose_path[patch_id] = img_path
                        self.all_image_paths.append(img_path)
                        self.all_label_paths.append(label_path)
                else:
                    self.all_image_paths.append(img_path)
                    self.all_label_paths.append(label_path)
        assert len(self.all_image_paths) == len(self.all_label_paths)
        
        patch_id_list = list(self.patch_id_to_canonical_pose_path.keys())
        for i in range(len(patch_id_list)):
            self.patch_id_to_patch_id_idx[patch_id_list[i]] = i

        # Will be set in prepare_dataset.py when train/val/test split is done.
        self.img_path_to_split = None
        self.img_path_by_patch_id_and_split = None

        # Store images/labels into np arrays for faster access.
        print('Loading images and labels into memory ...')
        self.all_images = np.array([load_image(path=img_path, target_dim=self.target_dim) for img_path in self.all_image_paths])
        self.all_labels = np.array([np.array(cv2.imread(label_path, cv2.IMREAD_UNCHANGED)) for label_path in self.all_label_paths])

        print(f'Finished. Images: {self.all_images.shape}; Labels: {self.all_labels.shape}\n')

        self.img_path_2_idx = {img_path: idx for idx, img_path in enumerate(self.all_image_paths)}

    def __len__(self) -> int:
        return len(self.all_image_paths)
    
    def __str__(self) -> str:
        return 'AugmentedMoNuSegDataset: %d images' % len(self)

    def __getitem__(self, idx) -> Tuple[np.array, np.array, str]:
        image_path = self.all_image_paths[idx]
        label_path = self.all_label_paths[idx]

        # image = load_image(path=image_path, target_dim=self.target_dim)
        # label = np.array(cv2.imread(label_path, cv2.IMREAD_UNCHANGED))
        image = self.all_images[idx]
        label = self.all_labels[idx]
        
        canonical_pose_image, canonical_pose_label, canonical_pose_img_path = self._canonical_pose(
            img_path=image_path)

        return (image, label, 
                canonical_pose_image, canonical_pose_label, 
                image_path, canonical_pose_img_path)
    
    def _canonical_pose(self, img_path) -> np.array:
        '''
            Return the canonical pose image/label of an image.
            For now, the mother of each augmented views is the canonical pose.
            e.g. 'TCGA-B0-5710-01Z-00-DX1_H172_W505_original.png' is the canonical pose of 
            'TCGA-NH-A8F7-01A-01-TS1_H673_W415_aug00001.png'
        '''
        patch_id = self.get_patch_id(img_path=img_path) # e.g. 'TCGA-NH-A8F7-01A-01-TS1_H673_W415'
        #print('patch_id:', patch_id)
        canonical_pose_img_path = self.patch_id_to_canonical_pose_path[patch_id]
        canonical_pose_label_path = canonical_pose_img_path.replace('image', 'label')

        # canonical_pose_img = load_image(path=canonical_pose_img_path, target_dim=self.target_dim)
        # canonical_pose_label = np.array(cv2.imread(canonical_pose_label_path, cv2.IMREAD_UNCHANGED))
        canonical_pose_img = self.all_images[self.img_path_2_idx[canonical_pose_img_path]]
        canonical_pose_label = self.all_labels[self.img_path_2_idx[canonical_pose_img_path]]

        return canonical_pose_img, canonical_pose_label, canonical_pose_img_path
    
    def get_patch_id(self, img_path) -> str:
        '''
            Return the patch_id of an image.
            '{path}/TCGA-NH-A8F7-01A-01-TS1_H673_W415_aug00001.png' -> 'TCGA-NH-A8F7-01A-01-TS1_H673_W415'
        '''
        img_file = os.path.basename(img_path)
        patch_id = "_".join(img_file.split('_')[:3])
        return patch_id
    
    def get_patch_id_idx(self, img_path) -> int:
        patch_id = self.get_patch_id(img_path)
        return self.patch_id_to_patch_id_idx[patch_id]

    def sample_views(self, split: str, patch_id: str, cnt: int = 1) -> Tuple[np.array, np.array]:
        '''
            Sample view(s) of the same patch id from the dataset.
            Similar to SimCLR sampling.
            Returns:
                images: [cnt, in_chan, W, H]
                labels: [cnt, W, H]
        '''
        images = []
        labels = []
        
        candiates = self.img_path_by_patch_id_and_split[patch_id][split]
        if cnt > len(candiates):
            raise ValueError('Not enough images for \
                             patch_id %s in split %s for %d views.' % (patch_id, split, cnt))
        idxs = np.random.randint(low=0, high=len(candiates), size=cnt)
        sampled_img_paths = [candiates[idx] for idx in idxs]

        for image_path in sampled_img_paths:
            label_path = image_path.replace('image', 'label')

            image = self.all_images[self.img_path_2_idx[image_path]]
            label = self.all_labels[self.img_path_2_idx[image_path]]
            # image = load_image(path=image_path, target_dim=self.target_dim)
            # label = np.array(cv2.imread(label_path, cv2.IMREAD_UNCHANGED))

            images.append(image[np.newaxis, ...])
            labels.append(label[np.newaxis, ...])
        
        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)

        return images, labels
    
    def _set_img_path_to_split(self, img_path_to_split: dict) -> None:
        '''
            Set the img_path_to_split dict
            Needed during sampling augmentation, to make sure no leakage between train/val/test sets.
        '''
        print('Setting img_path_to_split dict ...')

        self.img_path_to_split = img_path_to_split
        self.img_path_by_patch_id_and_split = {
            patch_id: {} for patch_id in self.patch_id_to_canonical_pose_path.keys()
        }

        for img_path, split in self.img_path_to_split.items():
            patch_id = self.get_patch_id(img_path=img_path) # e.g. 'TCGA-NH-A8F7-01A-01-TS1_H673_W415'
            if split not in self.img_path_by_patch_id_and_split[patch_id].keys():
                self.img_path_by_patch_id_and_split[patch_id][split] = [img_path]
            else:
                self.img_path_by_patch_id_and_split[patch_id][split].append(img_path)

        print('Finished setting img_path_to_split dict and img_path_by_patch_id_and_split.\n')        


def load_mask(path: str, target_dim: Tuple[int] = None) -> np.array:
    ''' Load mask as numpy array from a path string.'''
    mask = np.array(cv2.imread(path, cv2.IMREAD_UNCHANGED))

    if target_dim is not None:
        mask = np.array(cv2.resize(mask, target_dim))
    
    return mask # [W, H]


def load_image(path: str, target_dim: Tuple[int] = None) -> np.array:
    ''' Load image as numpy array from a path string.'''
    if target_dim is not None:
        image = np.array(
            cv2.resize(
                cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR),
                             code=cv2.COLOR_BGR2RGB), target_dim))
    else:
        image = np.array(
            cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR),
                         code=cv2.COLOR_BGR2RGB))

    # Normalize image.
    image = (image / 255 * 2) - 1

    # Channel last to channel first to comply with Torch.
    image = np.moveaxis(image, -1, 0)

    return image


if __name__ == '__main__':
    aug_lists = ['rotation',
                 'uniform_stretch',
                 'directional_stretch',
                 'volume_preserving_stretch',
                 'partial_stretch']
    
    dataset = AugmentedMoNuSegDataset(augmentation_methods=aug_lists)
    print(len(dataset))

    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

    for batch_idx, (images, labels, 
                canonical_pose_images, canonical_pose_labels, 
                image_paths, canonical_pose_img_paths) in enumerate(dataloader):
        print(images.shape)
        print(labels.shape)
        print(canonical_pose_images.shape)
        print(canonical_pose_labels.shape)
        #print(image_paths)
        #print(canonical_pose_img_paths)

        break

    