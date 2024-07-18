import cv2
import numpy as np


def augment_rotation(image: np.array,
                     label: np.array = None,
                     output_size: int = 64,
                     random_seed: int = None):
    '''
    Perform augmentation: rotation.
    The output image size can be smaller than the input image size.
    For simplicity, the input and output images/patches are always square shaped.
    '''
    input_size = image.shape[0]
    if output_size is None:
        output_size = input_size
    assert input_size >= output_size

    if random_seed is not None:
        np.random.seed(random_seed)

    angle = np.random.uniform(-180, 180)
    rotation_matrix = cv2.getRotationMatrix2D(
        (input_size / 2, input_size / 2), angle, 1)
    center_crop_hw_begin = (input_size -
                            output_size) // 2
    center_crop_hw_end = center_crop_hw_begin + output_size

    if len(image.shape) == 3:
        image_rotated = cv2.warpAffine(
            image, rotation_matrix,
            (input_size,
                input_size))[center_crop_hw_begin:center_crop_hw_end,
                            center_crop_hw_begin:center_crop_hw_end, :]
    else:
        assert len(image.shape) == 2
        image_rotated = cv2.warpAffine(
            image, rotation_matrix,
            (input_size,
                input_size))[center_crop_hw_begin:center_crop_hw_end,
                            center_crop_hw_begin:center_crop_hw_end]

    assert image_rotated.shape[0] == output_size
    assert image_rotated.shape[1] == output_size

    if label is not None:
        label_rotated = cv2.warpAffine(
            label,
            rotation_matrix, (input_size, input_size),
            flags=cv2.INTER_NEAREST)[
                center_crop_hw_begin:center_crop_hw_end,
                center_crop_hw_begin:center_crop_hw_end]

        assert label_rotated.shape[0] == output_size
        assert label_rotated.shape[1] == output_size

        return image_rotated, label_rotated
    return image_rotated
