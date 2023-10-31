import cv2
import numpy as np
from typing import Tuple


def augment_uniform_stretch(image: np.array,
                            label: np.array = None,
                            output_size: int = 64,
                            max_stretch_factor: float = 1.1,
                            can_squeeze: bool = False,
                            random_seed: int = None):
    '''
    Perform augmentation: uniform stretch.
    The output image size can be smaller than the input image size.
    For simplicity, the input and output images/patches are always square shaped.

    can_squeeze:
        If true, we can stretch or squeeze.
        Else, we can only stretch.
    '''
    input_size = image.shape[0]
    if output_size is None:
        output_size = input_size
    assert input_size >= output_size

    if random_seed is not None:
        np.random.seed(random_seed)

    assert max_stretch_factor >= 1.0
    stretch_factor = np.random.uniform(1.0, max_stretch_factor)
    if can_squeeze:
        # We do this to better randomize stretch/squeeze.
        # For example, if we randomize `stretched_size` with np.random.uniform(0.5, 2.0),
        # the result will be twice as likely to stretch than to squeeze.
        stretched_size = np.random.choice([
            int(input_size * stretch_factor),
            int(input_size / stretch_factor)
        ])
    else:
        stretched_size = int(input_size * stretch_factor)

    center_crop_hw_begin = (input_size - output_size) // 2
    center_crop_hw_end = center_crop_hw_begin + output_size

    if stretched_size >= input_size:
        placement_hw_begin = (stretched_size - input_size) // 2
        placement_hw_end = placement_hw_begin + input_size
        image_stretched = cv2.resize(image, (stretched_size, stretched_size))[
            placement_hw_begin:placement_hw_end,
            placement_hw_begin:placement_hw_end]
        if label is not None:
            label_stretched = cv2.resize(label, (stretched_size, stretched_size),
                                        interpolation=cv2.INTER_NEAREST)[
                                            placement_hw_begin:placement_hw_end,
                                            placement_hw_begin:placement_hw_end]
    else:
        placement_hw_begin = (input_size - stretched_size) // 2
        placement_hw_end = placement_hw_begin + stretched_size
        image_stretched = np.zeros_like(image)
        image_stretched[placement_hw_begin:placement_hw_end,
                        placement_hw_begin:placement_hw_end] = \
                            cv2.resize(image, (stretched_size, stretched_size))
        if label is not None:
            label_stretched = np.zeros_like(label)
            label_stretched[placement_hw_begin:placement_hw_end,
                            placement_hw_begin:placement_hw_end] = \
                                cv2.resize(label, (stretched_size, stretched_size),
                                        interpolation=cv2.INTER_NEAREST)

    if len(image.shape) == 3:
        image_stretched = image_stretched[center_crop_hw_begin:center_crop_hw_end,
                                          center_crop_hw_begin:center_crop_hw_end, :]
    else:
        assert len(image.shape) == 2
        image_stretched = image_stretched[center_crop_hw_begin:center_crop_hw_end,
                                          center_crop_hw_begin:center_crop_hw_end]

    assert image_stretched.shape[0] == output_size
    assert image_stretched.shape[1] == output_size

    if label is not None:
        label_stretched = label_stretched[
            center_crop_hw_begin:center_crop_hw_end,
            center_crop_hw_begin:center_crop_hw_end]

        assert label_stretched.shape[0] == output_size
        assert label_stretched.shape[1] == output_size

        return image_stretched, label_stretched
    return image_stretched


def augment_directional_stretch(image: np.array,
                                label: np.array = None,
                                output_size: int = 64,
                                max_stretch_factor: float = 1.1,
                                can_squeeze: bool = False,
                                random_seed: int = None):
    '''
    Perform augmentation: uniform stretch.
    The output image size can be smaller than the input image size.
    For simplicity, the input and output images/patches are always square shaped.

    can_squeeze:
        If true, we can stretch or squeeze.
        Else, we can only stretch.

    To implement directional stretch, we will
        1. Rotate the image at random angle.
        2. Stretch along the x-axis.
        3. Rotate it back.
    '''
    input_size = image.shape[0]
    if output_size is None:
        output_size = input_size
    assert input_size >= output_size

    if random_seed is not None:
        np.random.seed(random_seed)

    assert max_stretch_factor >= 1.0
    stretch_factor = np.random.uniform(1.0, max_stretch_factor)
    if can_squeeze:
        # We do this to better randomize stretch/squeeze.
        # For example, if we randomize `stretched_size` with np.random.uniform(0.5, 2.0),
        # the result will be twice as likely to stretch than to squeeze.
        stretched_size = np.random.choice([
            int(input_size * stretch_factor),
            int(input_size / stretch_factor)
        ])
    else:
        stretched_size = int(input_size * stretch_factor)

    center_crop_hw_begin = (input_size - output_size) // 2
    center_crop_hw_end = center_crop_hw_begin + output_size

    # Rotate forward.
    angle = np.random.uniform(-180, 180)
    rotation_matrix = cv2.getRotationMatrix2D(
        (input_size / 2, input_size / 2), angle, 1)
    image_rotFwd = cv2.warpAffine(
        image, rotation_matrix,
        (input_size, input_size))
    if label is not None:
        label_rotFwd = cv2.warpAffine(
            label,
            rotation_matrix, (input_size, input_size),
            flags=cv2.INTER_NEAREST)

    if stretched_size >= input_size:
        placement_h_begin = (stretched_size - input_size) // 2
        placement_h_end = placement_h_begin + input_size
        image_stretched = cv2.resize(image_rotFwd, (input_size, stretched_size))[
            placement_h_begin:placement_h_end, :]
        if label is not None:
            label_stretched = cv2.resize(label_rotFwd, (input_size, stretched_size),
                                        interpolation=cv2.INTER_NEAREST)[
                                            placement_h_begin:placement_h_end, :]
    else:
        placement_h_begin = (input_size - stretched_size) // 2
        placement_h_end = placement_h_begin + stretched_size
        image_stretched = np.zeros_like(image_rotFwd)
        image_stretched[placement_h_begin:placement_h_end, :] = \
            cv2.resize(image_rotFwd, (input_size, stretched_size))
        if label is not None:
            label_stretched = np.zeros_like(label_rotFwd)
            label_stretched[placement_h_begin:placement_h_end, :] = \
                                cv2.resize(label_rotFwd, (input_size, stretched_size),
                                        interpolation=cv2.INTER_NEAREST)

    # Rotate backward.
    angle = -angle
    rotation_matrix = cv2.getRotationMatrix2D(
        (input_size / 2, input_size / 2), angle, 1)
    image_rotBwd = cv2.warpAffine(
        image_stretched, rotation_matrix,
        (input_size, input_size))
    if label is not None:
        label_rotBwd = cv2.warpAffine(
            label_stretched,
            rotation_matrix, (input_size, input_size),
            flags=cv2.INTER_NEAREST)

    if len(image.shape) == 3:
        image_stretched = image_rotBwd[center_crop_hw_begin:center_crop_hw_end,
                                       center_crop_hw_begin:center_crop_hw_end, :]
    else:
        assert len(image.shape) == 2
        image_stretched = image_rotBwd[center_crop_hw_begin:center_crop_hw_end,
                                       center_crop_hw_begin:center_crop_hw_end]

    assert image_stretched.shape[0] == output_size
    assert image_stretched.shape[1] == output_size

    if label is not None:
        label_stretched = label_rotBwd[
            center_crop_hw_begin:center_crop_hw_end,
            center_crop_hw_begin:center_crop_hw_end]

        assert label_stretched.shape[0] == output_size
        assert label_stretched.shape[1] == output_size

        return image_stretched, label_stretched
    return image_stretched