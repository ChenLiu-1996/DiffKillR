import itertools
import os
import sys
from typing import Literal
from glob import glob
from typing import List, Tuple

import cv2
import pandas as pd
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/augmentation/')
from aug_rotation import augment_rotation
from aug_stretch import augment_uniform_stretch, augment_directional_stretch, augment_volume_preserving_stretch
from aug_partial_stretch import augment_partial_stretch
from center_crop import center_crop
sys.path.insert(0, import_dir + '/preprocessing/')
from Metas import MoNuSeg_Organ2FileID

ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])


class MoNuSegDataset(Dataset):
    def __init__(self,
                 augmentation_methods: List[str],
                 organ: str = None,
                 base_path: str = ROOT_DIR + '/data/MoNuSeg2018TrainData_patch_96x96/',
                 target_dim: Tuple[int] = (32, 32),
                 n_views: int = None):

        super().__init__()

        '''
        Find the list of relevant files.
        '''
        self.organ = organ
        self.target_dim = target_dim
        self.augmentation_methods = augmentation_methods
        self.n_views = n_views
        self.deterministic = False  # For infinite possibilities during training.

        self.img_paths = sorted(glob('%s/image/*.png' % base_path))
        self.label_paths = sorted(glob('%s/label/*.png' % base_path))

        if self.organ is not None:
            print('Organ: ', self.organ)
            organ_files = MoNuSeg_Organ2FileID[self.organ]['train'] # TOOD: may want to use test set as well.

            self.img_paths = [path for path in self.img_paths if any([file_id in path for file_id in organ_files])]
            self.label_paths = [path for path in self.label_paths if any([file_id in path for file_id in organ_files])]
        
        # Read class labelcsv file.
        self.class_labels_pd = pd.read_csv(f'{base_path}/class_labels.csv')
        self.patchID2Class = {row['patch_id']: row['type'] for _, row in self.class_labels_pd.iterrows()}
        self.num_cells = np.sum([self.patchID2Class[patchID] == 'cell' for patchID in self.patchID2Class])
        self.num_background = np.sum([self.patchID2Class[patchID] == 'background' for patchID in self.patchID2Class])
        print(f'Number of cells: {self.num_cells}')
        print(f'Number of background: {self.num_background}')
        print(f'Number of images: {len(self.img_paths)}')
        print(f'Number of labels: {len(self.label_paths)}')
        assert len(self.img_paths) == len(self.label_paths) + self.num_background

    def __len__(self) -> int:
        return len(self.img_paths)

    def __str__(self) -> str:
        return 'MoNuseg Dataset: %d images' % len(self)

    def set_deterministic(self, deterministic: bool):
        self.deterministic = deterministic

    def __getitem__(self, idx) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:

        # NOTE: we will not downsample the canonical images or labels.
        canonical_pose_image = load_image(path=self.img_paths[idx], target_dim=None)
        if idx < self.num_cells:
            # Load the label for the cell.
            canonical_pose_label = load_label(path=self.label_paths[idx], target_dim=None)
        else:
            # Load dummy label for background, since label is not used for DiffeoInvariantNet.
            canonical_pose_label = load_label(path=self.label_paths[self.num_cells-1], target_dim=None) # Dummy label to pass augmentation methods. This label will be ignored during training.

        if self.deterministic:
            # Set a fixed random seed for validation and testing.
            seed = idx + 1
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            # A handful of CUDA operations are nondeterministic if the CUDA version is
            # 10.2 or greater, unless the environment variable ``CUBLAS_WORKSPACE_CONFIG=:4096:8``
            # or ``CUBLAS_WORKSPACE_CONFIG=:16:8`` is set.
            os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
            torch.use_deterministic_algorithms(True)
            os.environ['PYTHONHASHSEED'] = str(seed)
        else:
            # TODO: How to unseed the other random generators?
            random.seed(None)
            np.random.seed(None)
            #torch.manual_seed(None)
            #torch.cuda.manual_seed(None)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.use_deterministic_algorithms(False)

        # Augment the image and label.
        augmentation_method = np.random.choice(self.augmentation_methods, size=1).item()
        aug_seed = np.random.randint(low=0, high=self.__len__() * 100)

        assert self.target_dim[0] == self.target_dim[1], \
            'AugmentedA28AxisDataset: currently only supporting square shape.'

        image_aug, label_aug = globals()['augment_' + augmentation_method](
            image=canonical_pose_image,
            label=canonical_pose_label,
            output_size=self.target_dim[0],
            random_seed=aug_seed,
        )[:2]

        # [1, C, H, W]
        image_aug = fix_channel_dimension(normalize_image(image_aug))
        label_aug = fix_channel_dimension(normalize_image(label_aug)) # axis label, need to be normalized.

        image_n_view, label_n_view = None, None
        if self.n_views is not None:
            # Prepare different views for contrastive learning purposes.
            image_n_view, label_n_view = [], []
            for _ in range(self.n_views):
                aug_seed = np.random.randint(low=0, high=self.__len__() * 100)
                augmentation_method = np.random.choice(self.augmentation_methods, size=1).item()
                image_new_view, label_new_view = globals()['augment_' + augmentation_method](
                    image=canonical_pose_image,
                    label=canonical_pose_label,
                    output_size=self.target_dim[0],
                    random_seed=aug_seed,
                )[:2]

                image_new_view = center_crop(image_new_view, output_size=self.target_dim[0])
                image_new_view = fix_channel_dimension(normalize_image(image_new_view))
                label_new_view = center_crop(label_new_view, output_size=self.target_dim[0])
                label_new_view = fix_channel_dimension(normalize_image(label_new_view))

                image_n_view.append(image_new_view[np.newaxis, ...])
                label_n_view.append(label_new_view[np.newaxis, ...])

            # [n_views, C, H, W]
            image_n_view = np.concatenate(image_n_view, axis=0)
            label_n_view = np.concatenate(label_n_view, axis=0)

        # Remember to center crop the canonical.
        canonical_pose_image = center_crop(canonical_pose_image, output_size=self.target_dim[0])
        canonical_pose_label = center_crop(canonical_pose_label, output_size=self.target_dim[0])

        canonical_pose_image = fix_channel_dimension(normalize_image(canonical_pose_image))
        canonical_pose_label = fix_channel_dimension(normalize_image(canonical_pose_label))

        #import pdb; pdb.set_trace()

        return (image_aug, label_aug,
                image_n_view, label_n_view,
                canonical_pose_image,
                canonical_pose_label)


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

    return image

def load_label(path: str, target_dim: Tuple[int] = None) -> np.array:
    ''' Load image as numpy array from a path string.'''

    if target_dim is not None:
        label = np.array(
            cv2.resize(cv2.imread(path, cv2.IMREAD_UNCHANGED), target_dim))
    else:
        label = np.array(
            cv2.imread(path, cv2.IMREAD_UNCHANGED))

    return label

def normalize_image(image: np.array) -> np.array:
    '''
    [0, 255] to [-1, 1]
    '''
    return image / 255.0 * 2 - 1

def fix_channel_dimension(arr: np.array) -> np.array:
    if len(arr.shape) == 3:
        # Channel last to channel first to comply with Torch.
        arr = np.moveaxis(arr, -1, 0)
    else:
        # Add channel dimension.
        assert len(arr.shape) == 2
        arr = arr[np.newaxis, ...]

    return arr


if __name__ == '__main__':
    aug_lists = ['rotation',
                 'uniform_stretch',
                 'directional_stretch',
                 'volume_preserving_stretch',
                 'partial_stretch']

    dataset = MoNuSeg(augmentation_methods=aug_lists, organ='Breast', target_dim=(32, 32), n_views=2)
    print(len(dataset))

    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

    for batch_idx, (images, _, image_n_view, _, canonical_images, _) in enumerate(dataloader):
        print(images.shape)
        print(image_n_view.shape)
        print(canonical_images.shape)
        break

