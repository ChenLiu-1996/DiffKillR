import os
import sys
import numpy as np
import argparse
from matplotlib import pyplot as plt
from typing import Tuple
from scipy.ndimage import center_of_mass

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir)

from test_augmentation.generate_shapes import generate_shape
from augmentation.aug_rotation import augment_rotation
from augmentation.aug_stretch import augment_uniform_stretch, augment_directional_stretch, augment_volume_preserving_stretch
from augmentation.aug_partial_stretch import augment_partial_stretch


def test_on_shape(shape: str = 'square'):
    fig = plt.figure(figsize=(10, 15))

    label = generate_shape(shape=shape, random_seed=1)
    image = np.zeros_like(label)[..., None].repeat(3, axis=-1)
    # Yale Blue color.
    image[:, :, 0] = 15/255*generate_shape(shape=shape, random_seed=1)
    image[:, :, 1] = 77/255*generate_shape(shape=shape, random_seed=1)
    image[:, :, 2] = 146/255*generate_shape(shape=shape, random_seed=1)

    ax = fig.add_subplot(10, 6, 1)
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title('Input "image"')
    ax = fig.add_subplot(10, 6, 7)
    ax.imshow(label, cmap='gray')
    ax.set_title('Input "label"')
    ax.set_axis_off()

    for seed in range(5):
        image_rotated, label_rotated = augment_rotation(image=image, label=label, output_size=64, random_seed=seed)
        ax = fig.add_subplot(10, 6, seed + 2)
        ax.imshow(image_rotated)
        ax.set_title('Rotation\n(seed %s)' % seed)
        ax.set_axis_off()
        ax = fig.add_subplot(10, 6, seed + 8)
        ax.imshow(label_rotated, cmap='gray')
        ax.set_title('Rotation\n(seed %s)' % seed)
        ax.set_axis_off()

    for seed in range(5):
        image_stretched, label_stretched = augment_uniform_stretch(image=image,
                                                                   label=label,
                                                                   output_size=64,
                                                                   max_stretch_factor=1.5,
                                                                   can_squeeze=True,
                                                                   random_seed=seed)
        ax = fig.add_subplot(10, 6, seed + 14)
        ax.imshow(image_stretched)
        ax.set_title('Uniform stretch\n(seed %s)' % seed)
        ax.set_axis_off()
        ax = fig.add_subplot(10, 6, seed + 20)
        ax.imshow(label_stretched, cmap='gray')
        ax.set_title('Uniform stretch\n(seed %s)' % seed)
        ax.set_axis_off()

    for seed in range(5):
        image_stretched, label_stretched, rotation_angle = \
            augment_directional_stretch(image=image,
                                        label=label,
                                        output_size=64,
                                        max_stretch_factor=1.5,
                                        can_squeeze=True,
                                        random_seed=seed)
        shape_centroid = center_of_mass(label)
        shape_centroid = shape_centroid[::-1]  # scipy H/W swapped.
        line_segment1 = find_line_with_point_and_angle(image.shape, shape_centroid, rotation_angle)
        line_segment2 = find_line_with_point_and_angle(image.shape, shape_centroid, rotation_angle + 180)
        ax = fig.add_subplot(10, 6, seed + 26)
        ax.imshow(image_stretched)
        ax.plot(*line_segment1, color='white', linestyle=':')
        ax.plot(*line_segment2, color='white', linestyle=':')
        ax.set_title('Directional stretch\n(seed %s)' % seed)
        ax.set_axis_off()
        ax = fig.add_subplot(10, 6, seed + 32)
        ax.imshow(label_stretched, cmap='gray')
        ax.set_title('Directional stretch\n(seed %s)' % seed)
        ax.set_axis_off()

    for seed in range(5):
        image_stretched, label_stretched, rotation_angle = \
            augment_volume_preserving_stretch(image=image,
                                              label=label,
                                              output_size=64,
                                              max_stretch_factor=1.5,
                                              random_seed=seed)

        shape_centroid = center_of_mass(label)
        shape_centroid = shape_centroid[::-1]  # scipy H/W swapped.
        line_segment1 = find_line_with_point_and_angle(image.shape, shape_centroid, rotation_angle)
        line_segment2 = find_line_with_point_and_angle(image.shape, shape_centroid, rotation_angle + 180)
        ax = fig.add_subplot(10, 6, seed + 38)
        ax.imshow(image_stretched)
        ax.plot(*line_segment1, color='white', linestyle=':')
        ax.plot(*line_segment2, color='white', linestyle=':')
        ax.set_title('Volume preserving\n(seed %s)' % seed)
        ax.set_axis_off()
        ax = fig.add_subplot(10, 6, seed + 44)
        ax.imshow(label_stretched, cmap='gray')
        ax.set_title('Volume preserving\n(seed %s)' % seed)
        ax.set_axis_off()

    for seed in range(5):
        image_stretched, label_stretched = \
            augment_partial_stretch(image=image,
                                    label=label,
                                    output_size=64,
                                    max_stretch_factor=1.5,
                                    random_seed=seed)

        ax = fig.add_subplot(10, 6, seed + 50)
        ax.imshow(image_stretched)
        ax.set_title('Partial stretch\n(seed %s)' % seed)
        ax.set_axis_off()
        ax = fig.add_subplot(10, 6, seed + 56)
        ax.imshow(label_stretched, cmap='gray')
        ax.set_title('Partial stretch\n(seed %s)' % seed)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig('synthetic_test_%s.png' % shape)


def find_line_with_point_and_angle(image_shape: Tuple[int], point: Tuple[int], angle: float):
    '''
    Find the two endpoints of a line segment spanning the entire image
    that passes through the given point at the given angle.

    `angle` is assumed to be in degrees.
    '''
    delta_h = np.cos(np.radians(angle))
    delta_w = np.sin(np.radians(angle))
    if delta_h > 0:
        max_length_h = (image_shape[0] - point[0]) / delta_h
    elif delta_h < 0:
        max_length_h = - point[0] / delta_h
    else:
        max_length_h = np.inf
    if delta_w > 0:
        max_length_w = (image_shape[1] - point[1]) / delta_w
    elif delta_w < 0:
        max_length_w = - point[1] / delta_w
    else:
        max_length_w = np.inf

    length_line = int(np.floor(min(max_length_h, max_length_w)))

    other_point = (int(point[0] + length_line * delta_h),
                   int(point[1] + length_line * delta_w))

    return [int(point[0]), other_point[0]], [int(point[1]), other_point[1]]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--shape',
        help='Shape to synthesize. One of `square`, `circle`, `triangle`, `ellipse`, `freeform`.',
        type=str,
        default='square')
    args = vars(parser.parse_args())
    test_on_shape(shape=args['shape'])