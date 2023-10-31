import os
import sys
import numpy as np
import argparse
from matplotlib import pyplot as plt

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir)

from synthetic.generate_shapes import generate_shape
from augmentation.aug_rotation import augment_rotation
from augmentation.aug_stretch import augment_uniform_stretch, augment_directional_stretch


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
        image_stretched, label_stretched = augment_directional_stretch(image=image,
                                                                       label=label,
                                                                       output_size=64,
                                                                       max_stretch_factor=1.5,
                                                                       can_squeeze=True,
                                                                       random_seed=seed)
        ax = fig.add_subplot(10, 6, seed + 26)
        ax.imshow(image_stretched)
        ax.set_title('Directional stretch\n(seed %s)' % seed)
        ax.set_axis_off()
        ax = fig.add_subplot(10, 6, seed + 32)
        ax.imshow(label_stretched, cmap='gray')
        ax.set_title('Directional stretch\n(seed %s)' % seed)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig('synthetic_test_%s.png' % shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--shape',
        help='Shape to synthesize. One of `square`, `circle`, `triangle`, `ellipse`, `freeform`.',
        type=str,
        default='square')
    args = vars(parser.parse_args())
    test_on_shape()