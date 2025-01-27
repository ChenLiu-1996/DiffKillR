import os
import sys
from glob import glob
from typing import List, Tuple
from simple_lama_inpainting import SimpleLama

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
sys.path.insert(0, import_dir + '/utils/')
from cell_isolation import isolate_cell, nonzero_value_closest_to_center

ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])


class GLySACDataset(Dataset):
    def __init__(self,
                 subset: str,
                 no_background: bool = False,
                 augmentation_methods: List[str] = None,
                 organ: str = 'Normal',
                 base_path: str = ROOT_DIR + '/data/GLySAC/GLySACbyTumor_patch_96x96/',
                 target_dim: Tuple[int] = (32, 32),
                 n_views: int = None,
                 percentage: int = 100,
                 cell_isolation: bool = False):

        super().__init__()

        '''
        Find the list of relevant files.
        '''
        self.subset = subset
        self.no_background = no_background
        self.organ = organ
        self.target_dim = target_dim
        self.augmentation_methods = augmentation_methods
        self.n_views = n_views
        self.percentage = percentage
        self.cell_isolation = cell_isolation
        if self.cell_isolation:
            self.inpainting_model = SimpleLama(device=torch.device('cpu'))
        self.deterministic = False  # For infinite possibilities during training.

        self.img_paths = sorted(glob(os.path.join(base_path, organ, subset, 'images', '*.png')))
        self.label_paths = sorted(glob(os.path.join(base_path, organ, subset, 'labels', '*.png')))
        self.background_img_paths = sorted(glob(os.path.join(base_path, organ, subset, 'background_images', '*.png')))

        background_ratio = len(self.background_img_paths) / len(self.img_paths)

        if self.subset == 'train':
            self.num_cells = int(np.floor(len(self.img_paths) * (self.percentage/100)))
            self.num_backgrounds = int(self.num_cells * background_ratio)
            print(f'Train set. Taking {self.percentage}% of the data: {self.num_cells}.')
            self.img_paths = self.img_paths[:self.num_cells]
            self.label_paths = self.label_paths[:self.num_cells]
            self.background_img_paths = self.background_img_paths[:self.num_backgrounds]
        else:
            self.num_cells = len(self.img_paths)
            self.num_backgrounds = len(self.background_img_paths)
            print(f'Test set. Taking all test data: {self.num_cells}.')

        assert len(self.img_paths) == len(self.label_paths) == self.num_cells
        assert len(self.background_img_paths) == self.num_backgrounds

    def __len__(self) -> int:
        if self.no_background:
            return self.num_cells
        else:
            return self.num_cells + self.num_backgrounds

    def __str__(self) -> str:
        return 'MoNuseg Dataset: %d images' % len(self)

    def set_deterministic(self, deterministic: bool):
        self.deterministic = deterministic

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        is_foreground = idx < self.num_cells

        if is_foreground:
            # NOTE: we will not downsample the canonical images or labels.
            canonical_pose_image = load_image(path=self.img_paths[idx], target_dim=None)
            # Load the label for the cell.
            canonical_pose_label = load_label(path=self.label_paths[idx], target_dim=None)
            if self.cell_isolation:
                # This image patch only contains the center cell.
                canonical_pose_image = isolate_cell(image=canonical_pose_image,
                                                    label=canonical_pose_label,
                                                    inpainting_model=self.inpainting_model)
            # This label patch only contains the center cell.
            center_cell_idx = nonzero_value_closest_to_center(canonical_pose_label)
            canonical_pose_label = canonical_pose_label == center_cell_idx
            canonical_pose_label = np.uint8(canonical_pose_label * 255)
        else:
            # NOTE: we will not downsample the canonical images or labels.
            canonical_pose_image = load_image(path=self.background_img_paths[idx - self.num_cells], target_dim=None)
            # Load dummy label for background, since label is not used for DiffeoInvariantNet.
            # Dummy label to pass augmentation methods. This label will be ignored during training.
            canonical_pose_label = np.ones_like(load_label(path=self.label_paths[0], target_dim=None))

        if canonical_pose_label.shape[-1] == 3:
            assert (canonical_pose_label[..., 0] == canonical_pose_label[..., 1]).all()
            assert (canonical_pose_label[..., 0] == canonical_pose_label[..., 2]).all()
            canonical_pose_label = canonical_pose_label[..., 0]

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
            'GLySACDataset: currently only supporting square shape.'

        image_aug, label_aug = globals()['augment_' + augmentation_method](
            image=canonical_pose_image,
            label=canonical_pose_label,
            output_size=self.target_dim[0],
            random_seed=aug_seed,
        )[:2]

        # [1, C, H, W]
        image_aug = fix_channel_dimension(normalize_image(image_aug))
        label_aug = fix_channel_dimension(normalize_label(label_aug))

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
                label_new_view = fix_channel_dimension(normalize_label(label_new_view))

                image_n_view.append(image_new_view[np.newaxis, ...])
                label_n_view.append(label_new_view[np.newaxis, ...])

            # [n_views, C, H, W]
            image_n_view = np.concatenate(image_n_view, axis=0)
            label_n_view = np.concatenate(label_n_view, axis=0)

        # Remember to center crop the canonical.
        canonical_pose_image = center_crop(canonical_pose_image, output_size=self.target_dim[0])
        canonical_pose_label = center_crop(canonical_pose_label, output_size=self.target_dim[0])

        canonical_pose_image = fix_channel_dimension(normalize_image(canonical_pose_image))
        canonical_pose_label = fix_channel_dimension(normalize_label(canonical_pose_label))

        return (image_aug, label_aug,
                image_n_view, label_n_view,
                canonical_pose_image,
                canonical_pose_label,
                is_foreground)


def load_image(path: str, target_dim: Tuple[int] = None) -> np.ndarray:
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

def load_label(path: str, target_dim: Tuple[int] = None) -> np.ndarray:
    ''' Load image as numpy array from a path string.'''

    if target_dim is not None:
        label = np.array(
            cv2.resize(cv2.imread(path, cv2.IMREAD_UNCHANGED), target_dim))
    else:
        label = np.array(
            cv2.imread(path, cv2.IMREAD_UNCHANGED))

    return label

def normalize_image(image: np.ndarray) -> np.ndarray:
    '''
    [0, 255] to [-1.0, 1.0]
    '''
    return image / 255.0 * 2 - 1

def normalize_label(label: np.ndarray) -> np.ndarray:
    '''
    [0, 255] to [0, 1]
    '''
    return np.uint8(label / 255)

def fix_channel_dimension(arr: np.ndarray) -> np.ndarray:
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

    dataset = GLySACDataset(augmentation_methods=aug_lists, organ='Normal', target_dim=(32, 32), n_views=2)
    print(len(dataset))

    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

    for batch_idx, (images, _, image_n_view, _, canonical_images, _) in enumerate(dataloader):
        print(images.shape)
        print(image_n_view.shape)
        print(canonical_images.shape)
        break

