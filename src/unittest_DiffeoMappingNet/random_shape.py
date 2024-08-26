from typing import Tuple, Optional
import os
import sys
import numpy as np
import torch
from matplotlib import pyplot as plt
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
from skimage.draw import polygon


import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/registration/')
from spatial_transformer import SpatialTransformer


def random_rectangle(image_size: Tuple[int] = (64, 64),
                     rectangle_size: Optional[Tuple[int]] = (32, 32),
                     center: Optional[Tuple[int]] = (32, 32),
                     random_seed: Optional[int] = 1):

    rectangle = np.zeros((*image_size, 3), dtype=np.uint8)

    if random_seed is not None:
        np.random.seed(random_seed)

    if rectangle_size is None:
        rectangle_size = np.array([
            np.random.randint(low=int(0.1 * image_size[0]), high=int(0.9 * image_size[0])),
            np.random.randint(low=int(0.1 * image_size[1]), high=int(0.9 * image_size[1])),
        ])

    if center is None:
        top_left_loc = np.array([
            np.random.randint(low=0, high=image_size[0] - rectangle_size[0]),
            np.random.randint(low=0, high=image_size[1] - rectangle_size[1]),
        ])
    else:
        top_left_loc = np.array([
            center[0] - rectangle_size[0] // 2,
            center[1] - rectangle_size[1] // 2,
        ])
    bot_right_loc = top_left_loc + np.array(rectangle_size)

    rectangle[top_left_loc[0]:bot_right_loc[0], top_left_loc[1]:bot_right_loc[1], :] = 255

    return rectangle


def random_star(image_size: Tuple[int] = (64, 64),
                center: Tuple[int] = (32, 32),
                num_points: int = 5):
    '''
    Currently not customizable.
    '''

    star = np.zeros((*image_size, 3), dtype=np.uint8)

    # Angles for the outer vertices
    outer_angles = np.linspace(np.pi, np.pi + 2 * np.pi, 2 * num_points + 1)[::2]
    # Angles for the inner vertices (shifted by half the angle between outer points)
    inner_angles = outer_angles + np.pi / num_points

    # Radius for the outer and inner vertices
    outer_radius = 24
    inner_radius = 9.1

    # Cartesian coordinates for outer and inner vertices
    outer_vertices = np.c_[outer_radius * np.cos(outer_angles),
                           outer_radius * np.sin(outer_angles)]
    inner_vertices = np.c_[inner_radius * np.cos(inner_angles),
                           inner_radius * np.sin(inner_angles)]

    # Combine the outer and inner vertices to form the pentagram
    vertices = np.vstack([outer_vertices[::2], outer_vertices[1::2], inner_vertices])
    rr, cc = polygon(center[0] + vertices[:, 0], center[1] + vertices[:, 1], star.shape)
    star[rr, cc, :] = 255

    return star


def random_triangle(image_size: Tuple[int] = (64, 64),
                    center: Tuple[int] = (32, 32),
                    radius: int = 24,
                    num_points: int = 3):
    '''
    Currently not customizable.
    '''

    triangle = np.zeros((*image_size, 3), dtype=np.uint8)

    # Angles for the outer vertices
    outer_angles = np.linspace(np.pi, np.pi + 2 * np.pi, 2 * num_points + 1)[::2]

    # Cartesian coordinates for vertices
    vertices = np.c_[radius * np.cos(outer_angles),
                     radius * np.sin(outer_angles)]

    # Combine the outer and inner vertices to form the pentagram
    rr, cc = polygon(center[0] + vertices[:, 0], center[1] + vertices[:, 1], triangle.shape)
    triangle[rr, cc, :] = 255

    return triangle


def register_dipy(moving_image: np.array, fixed_image: np.array):

    # Use classic registration to generate the warping field.
    metric = CCMetric(2, sigma_diff=2, radius=2)
    level_iters = [32, 16, 8]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
    mapping = sdr.optimize(fixed_image[..., 0], moving_image[..., 0])

    # The warp operation performed by Spatial Transformer.
    warper = SpatialTransformer(size=moving_image.shape[:2])
    warped_image = warper(torch.from_numpy(moving_image.transpose(2, 0, 1)[None, ...]).float(),
                          flow=torch.from_numpy(mapping.backward.transpose(2, 0, 1)[None, ...]))
    warped_image = np.uint8(warped_image[0, ...]).transpose(1, 2, 0)

    return warped_image, mapping.forward, mapping.backward


def plot_warp(fig, counter, moving_image, fixed_image):

    warped_image, diffeo_forward, _ = register_dipy(moving_image=moving_image, fixed_image=fixed_image)

    ax = fig.add_subplot(6, 4, counter * 4 + 1)
    ax.imshow(moving_image, vmin=0, vmax=255)
    ax.set_axis_off()
    ax.set_title('Moving Image', fontsize=16)

    ax = fig.add_subplot(6, 4, counter * 4 + 2)
    ax.imshow(fixed_image, vmin=0, vmax=255)
    ax.set_axis_off()
    ax.set_title('Fixed Image', fontsize=16)

    ax = fig.add_subplot(6, 4, counter * 4 + 3)
    vectors = [np.arange(0, s) for s in moving_image.shape[:2]]
    X, Y = np.meshgrid(vectors[0], vectors[1])
    warped_X = X + diffeo_forward[:, :, 1]
    warped_Y = Y + diffeo_forward[:, :, 0]
    for i in range(moving_image.shape[0]):
        ax.plot(warped_X[i, :], warped_Y[i, :], color='k')
    for j in range(moving_image.shape[1]):
        ax.plot(warped_X[:, j], warped_Y[:, j], color='k')
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title('Diffeomorphism (SyN)', fontsize=16)

    ax = fig.add_subplot(6, 4, counter * 4 + 4)
    ax.imshow(warped_image, vmin=0, vmax=255)
    ax.set_axis_off()
    ax.set_title('Warped Image (SyN)', fontsize=16)
    return


if __name__ == '__main__':
    rectangle = random_rectangle(rectangle_size=(32, 32), center=(32, 32))
    star = random_star(center=(32, 32))
    triangle = random_triangle(center=(36, 32))

    # Plot figure.
    plt.rcParams["font.family"] = 'serif'
    fig = plt.figure(figsize=(16, 24))

    plot_warp(fig, counter=0, moving_image=rectangle, fixed_image=star)
    plot_warp(fig, counter=1, moving_image=star, fixed_image=rectangle)
    plot_warp(fig, counter=2, moving_image=star, fixed_image=triangle)
    plot_warp(fig, counter=3, moving_image=triangle, fixed_image=star)
    plot_warp(fig, counter=4, moving_image=triangle, fixed_image=rectangle)
    plot_warp(fig, counter=5, moving_image=rectangle, fixed_image=triangle)

    fig.tight_layout(pad=2)
    fig.savefig('random_shape.png')

