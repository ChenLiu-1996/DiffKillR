import itertools
import os
import sys
from typing import Literal
from glob import glob
from typing import List, Tuple

import cv2
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

ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])


class AugmentedA28Dataset(Dataset):
    def __init__(self,
                 augmentation_methods: List[str],
                 cell_types: List[str] = ['EpithelialCell', 'EndothelialCell', 'Myocyte', 'Fibroblast'],
                 base_path: str = ROOT_DIR + '/data/A28-A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00_patch_96x96/',
                 target_dim: Tuple[int] = (32, 32),
                 n_views: int = None):

        super().__init__()

        '''
        Find the list of relevant files.
        '''

        self.target_dim = target_dim
        self.cell_types = cell_types
        self.cell_type_to_idx = {cell_type: idx for idx, cell_type in enumerate(self.cell_types)}
        self.augmentation_methods = augmentation_methods
        self.n_views = n_views
        self.deterministic = False  # For infinite possibilities during training.

        self.image_paths_by_celltype = {
            celltype: [] for celltype in self.cell_types
        }
        self.label_paths_by_celltype = {
            celltype: [] for celltype in self.cell_types
        }

        self.img_paths = sorted(glob('%s/image/*.png' % base_path))
        self.label_paths = sorted(glob('%s/label/*.png' % base_path))

        assert len(self.img_paths) == len(self.label_paths)

        for img_path, label_path in zip(self.img_paths, self.label_paths):
            file_name = img_path.split('/')[-1]
            celltype = file_name.split('_')[0] # e.g. 'EndotheliaCell'
            self.image_paths_by_celltype[celltype].append(img_path)
            self.label_paths_by_celltype[celltype].append(label_path)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __str__(self) -> str:
        return 'AugmentedDataset: %d images' % len(self)

    def set_deterministic(self, deterministic: bool):
        self.deterministic = deterministic

    def __getitem__(self, idx) -> Tuple[np.array, np.array, str]:

        # NOTE: we will not downsample the canonical images or labels.
        canonical_pose_image = load_image(path=self.img_paths[idx], target_dim=None)
        canonical_pose_label = load_label(path=self.label_paths[idx], target_dim=None)

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
            # torch.manual_seed and torch.cuda.manual_seed do not accept None Type.
            random.seed(None)
            np.random.seed(None)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.use_deterministic_algorithms(False)

        # Augment the image and label.
        augmentation_method = np.random.choice(self.augmentation_methods, size=1).item()
        aug_seed = np.random.randint(low=0, high=self.__len__() * 100)

        assert self.target_dim[0] == self.target_dim[1], \
            'AugmentedA28Dataset: currently only supporting square shape.'

        image_aug, label_aug = globals()['augment_' + augmentation_method](
            image=canonical_pose_image,
            label=canonical_pose_label,
            output_size=self.target_dim[0],
            random_seed=aug_seed,
        )[:2]

        # [1, C, H, W]
        image_aug = fix_channel_dimension(normalize_image(image_aug))
        label_aug = fix_channel_dimension(label_aug)

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

    dataset = AugmentedA28Dataset(augmentation_methods=aug_lists)
    print(len(dataset))

    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

    for batch_idx, (images, labels, image_paths) in enumerate(dataloader):
        print(images.shape)
        print(labels.shape)
        print(image_paths)
        break

