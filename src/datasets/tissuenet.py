import itertools
import os
from typing import Literal
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class TissueNetDataset(Dataset):
    def __init__(self,
                 base_path: str = '../../external_data/TissueNet/cellseg_partition/',
                 task: str = 'same_image_generalization',
                 target_dim: Tuple[int] = (32, 32)):

        super().__init__()

        self.base_path = base_path
        self.task = task
        self.target_dim = target_dim

        self.image_paths_by_celltype = {}
        self.mask_paths_by_celltype = {}

        celltype_list = [item.split('/')[-1] for item in glob('%s/%s/*' % (base_path, task))]

        for celltype in celltype_list:
            self.image_paths_by_celltype[celltype] = []
            self.mask_paths_by_celltype[celltype] = []

            folder = '%s/%s/%s/' % (base_path, task, celltype)
            img_paths = sorted(glob('%s/images/*.png' % (folder)))
            mask_paths = sorted(glob('%s/masks/*.png' % (folder)))
            for img_path, label_path in zip(img_paths, mask_paths):
                self.image_paths_by_celltype[celltype].append(img_path)
                self.mask_paths_by_celltype[celltype].append(label_path)

        self.all_image_paths = list(itertools.chain.from_iterable(self.image_paths_by_celltype.values()))
        self.all_mask_paths = list(itertools.chain.from_iterable(self.mask_paths_by_celltype.values()))

    def __len__(self) -> int:
        return len(self.all_image_paths)

    def __str__(self) -> str:
        return 'TissueNetDataset: %d images' % len(self)

    def __getitem__(self, idx) -> Tuple[np.array, np.array, str]:
        image_path = self.all_image_paths[idx]
        mask_path = self.all_mask_paths[idx]

        image = load_image(path=image_path, target_dim=self.target_dim)
        mask = np.array(cv2.imread(mask_path, cv2.IMREAD_UNCHANGED))
        other_image, other_mask, other_path = self._another_cell(img_path=1)

        return (image, mask, other_image, other_mask, image_path, other_path)

    def _another_cell(self, img_path) -> np.array:
        '''
        Get another cell from the same microscopy image.
        '''
        patch_id = self.get_patch_id(img_path=img_path)
        canonical_pose_img_path = self.patch_id_to_canonical_pose_path[patch_id]
        canonical_pose_label_path = canonical_pose_img_path.replace('image', 'label')

        canonical_pose_img = load_image(path=canonical_pose_img_path, target_dim=self.target_dim)
        canonical_pose_label = np.array(cv2.imread(canonical_pose_label_path, cv2.IMREAD_UNCHANGED))

        return canonical_pose_img, canonical_pose_label, canonical_pose_img_path

    def get_celltype(self, img_path) -> str:
        '''
        Return the celltype of an image.
        '''
        celltype = img_path.split('/')[-1].split('_')[0]
        return celltype

    def get_patch_id(self, img_path) -> str:
        '''
            Return the patch_id of an image.
        '''
        patch_id = "_".join(img_path.split('/')[-1].split('_')[:3])
        return patch_id

    def num_classes(self) -> int:
        return len(self.mask_paths_by_celltype.keys())

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
            image = load_image(path=image_path, target_dim=self.target_dim)
            label = np.array(cv2.imread(label_path, cv2.IMREAD_UNCHANGED))

            images.append(image[np.newaxis, ...])
            labels.append(label[np.newaxis, ...])

        images = np.concatenate(images, axis=0)
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
            label_path = image_path.replace('image', 'label')
            image = load_image(path=image_path, target_dim=self.target_dim) # [3, 96, 96]
            label = np.array(cv2.imread(label_path, cv2.IMREAD_UNCHANGED)) #[96, 96]

            images.append(image[np.newaxis, ...])
            labels.append(label[np.newaxis, ...])

        images = np.concatenate(images, axis=0)
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

