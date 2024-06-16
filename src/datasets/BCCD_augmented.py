'''
    Augmented BCCD dataset

'''

import itertools
import os
from typing import Literal
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from datasets.augmented import AugmentedDataset

class BCCD_augmentedDataset(AugmentedDataset):
    def __init__(self,
                augmentation_methods: List[str],
                cell_types: List[str] = ['WBC', 'RBC', 'Platelets'],
                base_path: str = '../data/BCCD_patch_augmented_128x128/train/',
                target_dim: Tuple[int] = (128, 128),
                has_labels: bool = False):
        super().__init__(augmentation_methods=augmentation_methods,
                        cell_types=cell_types,
                        base_path=base_path,
                        target_dim=target_dim,
                        has_labels=has_labels)
    
    def get_celltype(self, img_path) -> str:
        'BloodImage_00000_RBC_H0_W197_aug00000.png'
        #print('Overriding get_celltype() ...')
        img_name = os.path.basename(img_path)
        cell_type = img_name.split('_')[2]
        return cell_type
    
    def get_patch_id(self, img_path) -> str:
        'BloodImage_00000_RBC_H0_W197_aug00000.png -> BloodImage_00000_RBC'

        #print('Overriding get_patch_id() ...')

        img_name = os.path.basename(img_path)
        patch_id = "_".join(img_name.split('_')[:3])
        return patch_id
    

if __name__ == '__main__':
    aug_lists = ['rotation',
                 'uniform_stretch',
                 'directional_stretch',
                 'volume_preserving_stretch',
                 'partial_stretch']
    
    bccd_aug = BCCD_augmentedDataset(augmentation_methods=aug_lists)
    print(bccd_aug)

    dataloader = DataLoader(dataset=bccd_aug, batch_size=4, shuffle=True, num_workers=0)

    for batch_idx, (images, _, canonical_img, _, img_path, canonical_img_path) in enumerate(dataloader):
        print(images.shape)
        print(canonical_img)
        print(img_path)
        print(canonical_img_path)

        # sample by cell type
        # cell_type = 'RBC'
        # images, _ = bccd_aug.sample_celltype(split='train', celltype=cell_type, cnt=1)

        # print(images.shape)

        break
        
    

