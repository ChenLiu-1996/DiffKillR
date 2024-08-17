import itertools
import os
from typing import Literal
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class AugmentedDataset(Dataset):
    def __init__(self,
                 augmentation_methods: List[str],
                 cell_types: List[str] = ['EpithelialCell', 'EndothelialCell', 'Myocyte', 'Fibroblast'],
                 base_path: str = '../data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00_augmented_patch_32x32/',
                 target_dim: Tuple[int] = (32, 32),
                 has_labels: bool = True):

        super().__init__()

        self.target_dim = target_dim
        self.cell_types = cell_types
        self.base_path = base_path
        self.has_labels = has_labels

        self.cell_type_to_idx = {cell_type: idx for idx, cell_type in enumerate(self.cell_types)}
        self.augmentation_methods = augmentation_methods
        self.augmentation_folders = [
            folder for folder in sorted(glob('%s/*/' % base_path))
            if folder.split('/')[-2] in augmentation_methods
        ]
        print('Augmentation folders:', self.augmentation_folders)
        self.augmentation_method_to_folder = {}
        for folder in self.augmentation_folders:
            self.augmentation_method_to_folder[folder.split('/')[-2]] = folder

        self.image_paths_by_celltype = {
            celltype: [] for celltype in self.cell_types
        }
        if has_labels:
            self.label_paths_by_celltype = {
                celltype: [] for celltype in self.cell_types
            }
        
        # Note: make sure only 1 copy of patch_id_original.png exists in dataset
        # Otherwise, can't perform super contrastive learning.            
        self.patch_id_to_canonical_pose_path = {}
        self.patch_id_to_patch_id_idx = {}
        
        self._setup()

    def _setup(self) -> None:
        for folder in self.augmentation_folders:
            img_paths = sorted(glob('%s/image/*.png' % (folder)))
            #label_paths = sorted(glob('%s/label/*.png' % (folder)))
            for img_path in img_paths:
                #print('img_path:', img_path)
                file_name = os.path.basename(img_path)
                celltype = self.get_celltype(img_path=img_path)
                patch_id = self.get_patch_id(img_path=img_path)
                #celltype = file_name.split('_')[0] # e.g. 'EndotheliaCell'
                #patch_id = "_".join(file_name.split('_')[:3]) # e.g. 'EndotheliaCell_H7589_W9064'
                is_original = file_name.split('_')[-1].split('.')[0] == 'original'
                if is_original == True:
                    if patch_id not in self.patch_id_to_canonical_pose_path.keys():
                        self.patch_id_to_canonical_pose_path[patch_id] = img_path
                        self.image_paths_by_celltype[celltype].append(img_path)
                        # self.label_paths_by_celltype[celltype].append(label_path)
                else:
                    self.image_paths_by_celltype[celltype].append(img_path)
                    # self.label_paths_by_celltype[celltype].append(label_path)
        
        if self.has_labels:
            for folder in self.augmentation_folders:
                label_paths = sorted(glob('%s/label/*.png' % (folder)))
                for label_path in label_paths:
                    celltype = self.get_celltype(img_path=label_path)
                    if label_path not in self.label_paths_by_celltype[celltype]:
                        self.label_paths_by_celltype[celltype].append(label_path)

        patch_id_list = list(self.patch_id_to_canonical_pose_path.keys())
        for i in range(len(patch_id_list)):
            self.patch_id_to_patch_id_idx[patch_id_list[i]] = i

        self.all_image_paths = list(itertools.chain.from_iterable(self.image_paths_by_celltype.values()))
        print('Total images:', len(self.all_image_paths))
        if self.has_labels:
            self.all_label_paths = list(itertools.chain.from_iterable(self.label_paths_by_celltype.values()))
            assert len(self.all_image_paths) == len(self.all_label_paths)

        # Will be set in prepare_dataset.py when train/val/test split is done.
        self.img_path_to_split = None
        self.img_path_by_celltype_and_split = None
        self.img_path_by_patch_id_and_split = None

        print('Finished setting up AugmentedDataset.')


    def __len__(self) -> int:
        return len(self.all_image_paths)
    
    def __str__(self) -> str:
        return 'AugmentedDataset: %d images' % len(self)

    def __getitem__(self, idx) -> Tuple[np.array, np.array, str]:
        image_path = self.all_image_paths[idx]
        if self.has_labels:
            label_path = self.all_label_paths[idx]

        image = load_image(path=image_path, target_dim=self.target_dim)
        if self.has_labels:
            label = np.array(cv2.imread(label_path, cv2.IMREAD_UNCHANGED))
        canonical_pose_image, canonical_pose_label, canonical_pose_img_path = self._canonical_pose(
            img_path=image_path)

        #return image, label, canonical_pose_image, canonical_pose_label, image_path
        if self.has_labels:
            return (image, label, 
                    canonical_pose_image, canonical_pose_label, 
                    image_path, canonical_pose_img_path)
        else:
            return (image, np.empty(0),
                    canonical_pose_image, np.empty(0),
                    image_path, canonical_pose_img_path)
    
    def _canonical_pose(self, img_path) -> np.array:
        '''
            Return the canonical pose image/label of a celltype.
            For now, the mother of each augmented views is the canonical pose.
            e.g. EndothelialCell_H7589_W9064_original.png, is the canonical pose of 
            EndothelialCell_H7589_W9064_*.png
        '''
        patch_id = self.get_patch_id(img_path=img_path)
        canonical_pose_img_path = self.patch_id_to_canonical_pose_path[patch_id]
        if self.has_labels:
            canonical_pose_label_path = canonical_pose_img_path.replace('image', 'label')

        canonical_pose_img = load_image(path=canonical_pose_img_path, target_dim=self.target_dim)
        if self.has_labels:
            canonical_pose_label = np.array(cv2.imread(canonical_pose_label_path, cv2.IMREAD_UNCHANGED))

        if self.has_labels:
            return canonical_pose_img, canonical_pose_label, canonical_pose_img_path
        else:
            return canonical_pose_img, None, canonical_pose_img_path
    
    def get_celltype(self, img_path) -> str:
        '''
            Return the celltype of an image.
            Can be overridden if the celltype is not the first part of the filename.
        '''
        celltype = img_path.split('/')[-1].split('_')[0]
        return celltype
    
    def get_patch_id(self, img_path) -> str:
        '''
            Return the patch_id of an image.
            Can be overridden if the patch_id is not the first 3 parts of the filename.
        '''
        patch_id = "_".join(img_path.split('/')[-1].split('_')[:3])
        return patch_id
    
    def get_patch_id_idx(self, img_path) -> int:
        patch_id = self.get_patch_id(img_path)
        return self.patch_id_to_patch_id_idx[patch_id]
    
    def get_celltype_id(self, cell_type) -> int:
        cell_list = list(self.image_paths_by_celltype.keys())
        return cell_list.index(cell_type)

    def num_classes(self) -> int:
        return len(self.image_paths_by_celltype.keys())

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
            image = load_image(path=image_path, target_dim=self.target_dim)
            images.append(image[np.newaxis, ...])

            if self.has_labels:
                label_path = image_path.replace('image', 'label')
                label = np.array(cv2.imread(label_path, cv2.IMREAD_UNCHANGED))
                labels.append(label[np.newaxis, ...])


        images = np.concatenate(images, axis=0)
        if self.has_labels:
            labels = np.concatenate(labels, axis=0)

        return images, labels
       
    
    def sample_celltype(self, split: str, celltype: str, cnt: int = 1) -> Tuple[np.array, np.array]:
        '''
            Sample image, label with a specific celltype from the dataset.
            Returns:
                images: [cnt, in_chan, W, H]
                labels: [cnt, W, H]
        '''
        if celltype not in self.cell_types:
            raise ValueError('Celltype %s not found in the dataset.' % celltype)
        
        images = []
        labels = []
        
        candiates = self.img_path_by_celltype_and_split[celltype][split]
        if cnt > len(candiates):
            raise ValueError('Not enough images for \
                             celltype %s in split %s for %d views.' % (celltype, split, cnt))
        idxs = np.random.randint(low=0, high=len(candiates), size=cnt)
        sampled_img_paths = [candiates[idx] for idx in idxs]

        for image_path in sampled_img_paths:
            image = load_image(path=image_path, target_dim=self.target_dim) # [3, 96, 96]
            images.append(image[np.newaxis, ...]) 
            if self.has_labels:
                label_path = image_path.replace('image', 'label')
                label = np.array(cv2.imread(label_path, cv2.IMREAD_UNCHANGED)) #[96, 96]
                labels.append(label[np.newaxis, ...])
            
        images = np.concatenate(images, axis=0)
        if self.has_labels:
            labels = np.concatenate(labels, axis=0)

        return images, labels
    
    def _set_img_path_to_split(self, img_path_to_split: dict) -> None:
        '''
            Set the img_path_to_split dict & img_path_by_celltype_and_split dict.
            Needed during sampling augmentation, to make sure no leakage between train/val/test sets.
        '''
        print('Setting img_path_to_split dict and img_path_by_celltype_and_split ...')

        self.img_path_to_split = img_path_to_split
        self.img_path_by_celltype_and_split = {
            celltype: {} for celltype in self.cell_types
        }
        self.img_path_by_patch_id_and_split = {
            patch_id: {} for patch_id in self.patch_id_to_canonical_pose_path.keys()
        }

        for img_path, split in self.img_path_to_split.items():
            celltype = self.get_celltype(img_path=img_path)
            if split not in self.img_path_by_celltype_and_split[celltype].keys():
                self.img_path_by_celltype_and_split[celltype][split] = [img_path]
            else:
                self.img_path_by_celltype_and_split[celltype][split].append(img_path)
            
            patch_id = self.get_patch_id(img_path=img_path) # e.g. 'EndotheliaCell_H7589_W9064'
            if split not in self.img_path_by_patch_id_and_split[patch_id].keys():
                self.img_path_by_patch_id_and_split[patch_id][split] = [img_path]
            else:
                self.img_path_by_patch_id_and_split[patch_id][split].append(img_path)

        print('Finished setting img_path_to_split dict; \
              img_path_by_celltype_and_split; img_path_by_patch_id_and_split.\n')        


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
    
    dataset = AugmentedDataset(augmentation_methods=aug_lists)
    print(len(dataset))

    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

    for batch_idx, (images, labels, image_paths) in enumerate(dataloader):
        print(images.shape)
        print(labels.shape)
        print(image_paths)
        break

    