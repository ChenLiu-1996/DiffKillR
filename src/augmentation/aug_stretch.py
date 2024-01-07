import cv2
import numpy as np
from typing import Tuple


def augment_uniform_stretch(image: np.array,
                            label: np.array = None,
                            output_size: int = 64,
                            max_stretch_factor: float = 1.5,
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
                                max_stretch_factor: float = 1.5,
                                can_squeeze: bool = False,
                                random_seed: int = None):
    '''
    Perform augmentation: directional stretch.
    The output image size can be smaller than the input image size.
    For simplicity, the input and output images/patches are always square shaped.

    can_squeeze:
        If true, we can stretch or squeeze.
        Else, we can only stretch.

    To implement directional stretch, we will
        1. Rotate the image at random angle.
        2. Stretch along the h-axis.
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
    # Report the angle for rotation for plotting purposes.
    rotation_angle = angle

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

        return image_stretched, label_stretched, rotation_angle
    return image_stretched, rotation_angle

def augment_volume_preserving_stretch(image: np.array,
                                      label: np.array = None,
                                      output_size: int = 64,
                                      max_stretch_factor: float = 1.5,
                                      random_seed: int = None):
    '''
    Perform volume preserving stretch.
    The output image size can be smaller than the input image size.
    For simplicity, the input and output images/patches are always square shaped.

    To implement volume preserving stretch, we will
        1. Find the principal axes of the object.
        2. Rotate the image clockwise by the angle between the x-axis and the first principal axis.
        2. Stretch along one principal axes and shrink along the other. Randomly decide which one to stretch and which one to shrink.
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
    stretched_size = int(input_size * stretch_factor)

    center_crop_hw_begin = (input_size - output_size) // 2
    center_crop_hw_end = center_crop_hw_begin + output_size

    # Find the principal axes of the object.
    _, _, eigenvectors = _get_centroid_and_axes(image)
    paxis1 = eigenvectors[0,:]
    paxis2 = eigenvectors[1,:]

    # Angle between the x-axis and the first principal axis.
    radian = np.arccos(np.dot(paxis1, np.array([1,0])) / (np.linalg.norm(paxis1) * 1))
    angle = np.degrees(radian)
    # Report the angle for rotation for plotting purposes.
    rotation_angle = angle

    # Rotate forward.
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


    # stretched_size > input_size, always.
    placement_h_begin = (stretched_size - input_size) // 2
    placement_h_end = placement_h_begin + input_size

    # Randomly decide which principal axis to stretch and which one to shrink.
    # TODO: !check whether placement_h_begin and placement_h_end are correct.
    if np.random.uniform(0, 1) > 0.5:
        image_stretched = cv2.resize(image_rotFwd, (input_size, stretched_size))[
            placement_h_begin:placement_h_end, :]
        if label is not None:
            label_stretched = cv2.resize(label_rotFwd, (input_size, stretched_size),
                                        interpolation=cv2.INTER_NEAREST)[
                                            placement_h_begin:placement_h_end, :]
    else:
        image_stretched = cv2.resize(image_rotFwd, (stretched_size, input_size))[
            :, placement_h_begin:placement_h_end]
        if label is not None:
            label_stretched = cv2.resize(label_rotFwd, (stretched_size, input_size),
                                        interpolation=cv2.INTER_NEAREST)[
                                            :, placement_h_begin:placement_h_end]

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

        return image_stretched, label_stretched, rotation_angle
    return image_stretched, rotation_angle



def _get_centroid_and_axes(image: np.array) -> Tuple[Tuple[int, int], np.array, np.array]:
    '''
        Return centroid and principal axes of object in image.
        image: numpy array of shape (height, width, channel)
        centroid: tuple of (x,y) coordinates of centroid
        eigenvalues: tuple of eigenvalues of covariance matrix
        eigenvectors: tuple of eigenvectors of covariance matrix

        Note: The eigenvalues do not reflect the actual length of the principal axes, but
        rather the spread of pixel values along the principal axes.

    '''
    # Convert image to grayscale
    if image.shape[-1] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Calculate Moments
    moments = cv2.moments(gray)

    # Calculate x,y coordinate of center
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])

    # Calculate the covariance matrix
    mu20 = moments['mu20'] / moments['m00']
    mu02 = moments['mu02'] / moments['m00']
    mu11 = moments['mu11'] / moments['m00']

    covariance_matrix = np.array([[mu20, mu11], [mu11, mu02]])

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    return (cX, cY), eigenvalues, eigenvectors


if __name__ == '__main__':
    pass