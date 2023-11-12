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
                 base_path: str = '../data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00_augmented_patch_96x96/',
                 target_dim: Tuple[int] = (96, 96)):

        super().__init__()

        self.target_dim = target_dim
        self.cell_types = cell_types
        self.augmentation_methods = augmentation_methods
        self.augmentation_folders = [
            folder for folder in sorted(glob('%s/*/' % base_path))
            if folder.split('/')[-2] in augmentation_methods
        ]
        
        self.augmentation_folders_map = {}
        for folder in self.augmentation_folders:
            self.augmentation_folders_map[folder.split('/')[-2]] = folder

        self.image_paths_by_celltype = {
            celltype: [] for celltype in self.cell_types
        }
        self.label_paths_by_celltype = {
            celltype: [] for celltype in self.cell_types
        }
        
        # Note: make sure only 1 copy of patch_id_original.png exists in dataset
        # Otherwise, can't perform super contrastive learning.            
        self.patch_id_to_canonical_pose_path = {}
        original_cnt = 0
        og_dup_cnt = 0
        aug_cnt = 0

        for folder in self.augmentation_folders:
            img_paths = sorted(glob('%s/image/*.png' % (folder)))
            label_paths = sorted(glob('%s/label/*.png' % (folder)))
            for img_path, label_path in zip(img_paths, label_paths):
                print('img_path: ', img_path)
                print('label_path: ', label_path)
                file_name = img_path.split('/')[-1]
                celltype = file_name.split('_')[0] # e.g. 'EndotheliaCell'
                patch_id = "_".join(file_name.split('_')[:3]) # e.g. 'EndotheliaCell_H7589_W9064'
                is_original = file_name.split('_')[-1].split('.')[0] == 'original'
                print('celltype: ', celltype, 'patch_id: ', patch_id, 'is_original: ', is_original)
                if is_original == True:
                    original_cnt += 1
                    if patch_id not in self.patch_id_to_canonical_pose_path.keys():
                        self.patch_id_to_canonical_pose_path[patch_id] = img_path
                        self.image_paths_by_celltype[celltype].append(img_path)
                        self.label_paths_by_celltype[celltype].append(label_path)
                    else:
                        og_dup_cnt += 1
                else:
                    aug_cnt += 1
                    self.image_paths_by_celltype[celltype].append(img_path)
                    self.label_paths_by_celltype[celltype].append(label_path)
                print('-----------------------\n')
        #print('augmentation_folders_map: ')
        #print(self.augmentation_folders_map)
        print('original_cnt: ', original_cnt, 'aug_cnt: ', aug_cnt, 'og_dup_cnt: ', og_dup_cnt)

        self.all_image_paths = list(itertools.chain.from_iterable(self.image_paths_by_celltype.values()))
        self.all_label_paths = list(itertools.chain.from_iterable(self.label_paths_by_celltype.values()))
        print('self.all_image_paths: ', self.all_image_paths[:10])

        assert len(self.all_image_paths) == len(self.all_label_paths)


    def __len__(self) -> int:
        return len(self.all_image_paths)

    def __getitem__(self, idx) -> Tuple[np.array, np.array, str]:
        image_path = self.all_image_paths[idx]
        label_path = self.all_label_paths[idx]

        image = load_image(path=image_path, target_dim=self.target_dim)
        label = np.array(cv2.imread(label_path, cv2.IMREAD_UNCHANGED))


        return image, label, image_path
    
    def _canonical_pose(self, img_path) -> np.array:
        '''
            Return the canonical pose of a celltype.
            For now, the mother of each augmented views is the canonical pose.
            e.g. EndothelialCell_H7589_W9064_original.png, is the canonical pose of EndothelialCell_H7589_W9064_augxxxx.png
        '''
        patch_id = "_".join(img_path.split('/')[-1].split('_')[:3])
        canonical_pose_path = self.patch_id_to_canonical_pose_path[patch_id]

        canonical_pose = load_image(path=canonical_pose_path, target_dim=self.target_dim)

        return canonical_pose

    def num_classes(self) -> int:
        return len(self.label_paths_by_celltype.keys())
    
    def sample_celltype(self, celltype: str) -> Tuple[np.array, np.array]:
        '''
            Sample image, label with a specific celltype from the dataset.
    
        '''
        if celltype not in self.label_paths_by_celltype.keys():
            raise ValueError('Celltype %s not found in the dataset.' % celltype)
        
        idx = np.random.randint(0, len(self.label_paths_by_celltype[celltype]))
        image_path = self.image_paths_by_celltype[celltype][idx]
        label_path = self.label_paths_by_celltype[celltype][idx]

        image = load_image(path=image_path, target_dim=self.target_dim)
        label = np.array(cv2.imread(label_path, cv2.IMREAD_UNCHANGED))

        return image, label
        


def load_image(path: str, target_dim: Tuple[int] = None) -> np.array:
    ''' Load image as numpy array from a path string.'''
    print('path: ', path)
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

    