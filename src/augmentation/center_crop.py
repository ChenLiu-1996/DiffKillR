'''
Strictly speaking, this files does not perform any augmentation.
This is a utility file for image processing.
'''
import numpy as np


def center_crop(image: np.array,
                label: np.array = None,
                output_size: int = 64):
    '''
    Crop the image and label at the center.
    '''
    input_size = image.shape[0]
    if output_size is None:
        output_size = input_size
    assert input_size >= output_size

    center_crop_hw_begin = (input_size -
                            output_size) // 2
    center_crop_hw_end = center_crop_hw_begin + output_size

    if len(image.shape) == 3:
        image = image[center_crop_hw_begin:center_crop_hw_end,
                      center_crop_hw_begin:center_crop_hw_end, :]
    else:
        assert len(image.shape) == 2
        image = image[center_crop_hw_begin:center_crop_hw_end,
                      center_crop_hw_begin:center_crop_hw_end]

    assert image.shape[0] == output_size
    assert image.shape[1] == output_size

    if label is not None:
        label = label[center_crop_hw_begin:center_crop_hw_end,
                      center_crop_hw_begin:center_crop_hw_end]

        assert label.shape[0] == output_size
        assert label.shape[1] == output_size

        return image, label
    return image
