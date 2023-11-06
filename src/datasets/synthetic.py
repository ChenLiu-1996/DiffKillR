import itertools
import os
from typing import Literal
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):

    def __init__(self,
                 base_path: str = '../../data/synthesized/',
                 target_dim: Tuple[int] = (64, 64)):

        super().__init__()

        self.target_dim = target_dim
        all_image_folders = sorted(glob('%s/*/' % base_path))

        self.image_by_celltype = []

        for folder in all_image_folders:
            paths = sorted(glob('%s/*.png' % (folder)))
            if len(paths) >= 2:
                self.image_by_celltype.append(paths)

    def __len__(self) -> int:
        return len(self.image_by_celltype[0])

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        #TODO: Think about how to batch things for SupContrastive learning.

        return images, labels


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
