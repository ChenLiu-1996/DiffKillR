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
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(5, 6, 1)
    ax.set_axis_off()

    image = generate_shape(shape=shape, random_seed=1)
    ax.imshow(image, cmap='gray')
    ax.set_title('Input shape')

    for seed in range(5):
        ax = fig.add_subplot(5, 6, seed + 2)
        image_rotated = augment_rotation(image=image, output_size=64, random_seed=seed)
        ax.imshow(image_rotated, cmap='gray')
        ax.set_title('Rotation\n(seed %s)' % seed)
        ax.set_axis_off()

    for seed in range(5):
        ax = fig.add_subplot(5, 6, seed + 8)
        image_stretched = augment_uniform_stretch(image=image,
                                                  output_size=64,
                                                  max_stretch_factor=1.5,
                                                  can_squeeze=True,
                                                  random_seed=seed)
        ax.imshow(image_stretched, cmap='gray')
        ax.set_title('Uniform stretch\n(seed %s)' % seed)
        ax.set_axis_off()

    for seed in range(5):
        ax = fig.add_subplot(5, 6, seed + 14)
        image_stretched = augment_directional_stretch(image=image,
                                                      output_size=64,
                                                      max_stretch_factor=1.5,
                                                      can_squeeze=True,
                                                      random_seed=seed)
        ax.imshow(image_stretched, cmap='gray')
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