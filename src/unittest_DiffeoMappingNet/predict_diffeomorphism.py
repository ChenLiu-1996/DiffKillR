import os
import sys
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from random_shape import register_dipy, random_rectangle, random_triangle, random_star


import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/registration/')
from unet import UNet
from voxelmorph import VxmDense as VoxelMorph
from spatial_transformer import SpatialTransformer


def plot_predict_warp(fig, counter, moving_image, fixed_image):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    warped_image_dipy, diffeo_forward_dipy, _ = register_dipy(moving_image=moving_image, fixed_image=fixed_image)

    ax = fig.add_subplot(6, 7, counter * 7 + 1)
    ax.imshow(moving_image, vmin=0, vmax=255)
    ax.set_axis_off()
    ax.set_title('Original Image', fontsize=16)

    ax = fig.add_subplot(6, 7, counter * 7 + 2)
    ax.imshow(warped_image_dipy, vmin=0, vmax=255)
    ax.set_axis_off()
    ax.set_title('Warped Image', fontsize=16)

    ax = fig.add_subplot(6, 7, counter * 7 + 3)
    vectors = [np.arange(0, s) for s in moving_image.shape[:2]]
    X, Y = np.meshgrid(vectors[0], vectors[1])
    warped_X = X + diffeo_forward_dipy[:, :, 1]
    warped_Y = Y + diffeo_forward_dipy[:, :, 0]
    for i in range(moving_image.shape[0]):
        ax.plot(warped_X[i, :], warped_Y[i, :], color='k')
    for j in range(moving_image.shape[1]):
        ax.plot(warped_X[:, j], warped_Y[:, j], color='k')
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title('Ground Truth Diffeomorphism', fontsize=16)

    # NOTE: DiffeoMappingNet with UNet.
    DiffeoMappingNet = UNet(
        num_filters=32,
        in_channels=6,
        out_channels=4)
    DiffeoMappingNet = DiffeoMappingNet.to(device)
    DiffeoMappingNet.train()
    warper = SpatialTransformer(size=moving_image.shape[:2])
    warper = warper.to(device)

    optimizer = torch.optim.AdamW(DiffeoMappingNet.parameters(), lr=5e-4)
    mse_loss = torch.nn.MSELoss()

    moving_image_torch = torch.from_numpy((moving_image).transpose(2, 0, 1)[None, ...]).float()
    fixed_image_torch = torch.from_numpy((fixed_image).transpose(2, 0, 1)[None, ...]).float()

    for _ in tqdm(range(100)):
        __diffeo_forward, __diffeo_backward = DiffeoMappingNet(source=moving_image_torch, target=fixed_image_torch)
        image_warped_forward = warper(moving_image_torch, flow=__diffeo_forward)
        image_warped_backward = warper(fixed_image_torch, flow=__diffeo_backward)

        loss_forward = mse_loss(fixed_image_torch, image_warped_forward)
        loss_cyclic = mse_loss(moving_image_torch, image_warped_backward)
        loss = loss_forward + loss_cyclic
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    DiffeoMappingNet.eval()
    diffeo_forward_unet, _ = DiffeoMappingNet(source=moving_image_torch, target=fixed_image_torch)
    diffeo_forward_unet = diffeo_forward_unet.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)
    warped_image_unet = warper(torch.from_numpy(moving_image.transpose(2, 0, 1)[None, ...]).float(),
                               flow=torch.from_numpy(diffeo_forward_unet.transpose(2, 0, 1)[None, ...]))
    warped_image_unet = np.uint8(warped_image_unet[0, ...]).transpose(1, 2, 0)

    ax = fig.add_subplot(6, 7, counter * 7 + 4)
    ax.imshow(warped_image_unet, vmin=0, vmax=255)
    ax.set_axis_off()
    ax.set_title('Predicted Warped Image\nDiffeoMappingNet (UNet)', fontsize=16)

    ax = fig.add_subplot(6, 7, counter * 7 + 5)
    vectors = [np.arange(0, s) for s in moving_image.shape[:2]]
    X, Y = np.meshgrid(vectors[0], vectors[1])
    warped_X = X + diffeo_forward_unet[:, :, 1]
    warped_Y = Y + diffeo_forward_unet[:, :, 0]
    for i in range(moving_image.shape[0]):
        ax.plot(warped_X[i, :], warped_Y[i, :], color='k')
    for j in range(moving_image.shape[1]):
        ax.plot(warped_X[:, j], warped_Y[:, j], color='k')
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title('Predicted Diffeomorphism\nDiffeoMappingNet (UNet)', fontsize=16)

    # NOTE: DiffeoMappingNet with VoxelMorph.
    DiffeoMappingNet = VoxelMorph(
        inshape=(64, 64),
        src_feats=3,
        trg_feats=3,
        bidir=True)
    DiffeoMappingNet = DiffeoMappingNet.to(device)
    DiffeoMappingNet.train()
    warper = SpatialTransformer(size=moving_image.shape[:2])
    warper = warper.to(device)

    optimizer = torch.optim.AdamW(DiffeoMappingNet.parameters(), lr=5e-4)
    mse_loss = torch.nn.MSELoss()

    moving_image_torch = torch.from_numpy((moving_image).transpose(2, 0, 1)[None, ...]).float()
    fixed_image_torch = torch.from_numpy((fixed_image).transpose(2, 0, 1)[None, ...]).float()

    for _ in tqdm(range(100)):
        __diffeo_forward, __diffeo_backward = DiffeoMappingNet(source=moving_image_torch, target=fixed_image_torch)
        image_warped_forward = warper(moving_image_torch, flow=__diffeo_forward)
        image_warped_backward = warper(fixed_image_torch, flow=__diffeo_backward)

        loss_forward = mse_loss(fixed_image_torch, image_warped_forward)
        loss_cyclic = mse_loss(moving_image_torch, image_warped_backward)
        loss = loss_forward + loss_cyclic
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    DiffeoMappingNet.eval()
    diffeo_forward_voxelmorph, _ = DiffeoMappingNet(source=moving_image_torch, target=fixed_image_torch)
    diffeo_forward_voxelmorph = diffeo_forward_voxelmorph.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)
    warped_image_voxelmorph = warper(torch.from_numpy(moving_image.transpose(2, 0, 1)[None, ...]).float(),
                               flow=torch.from_numpy(diffeo_forward_voxelmorph.transpose(2, 0, 1)[None, ...]))
    warped_image_voxelmorph = np.uint8(warped_image_voxelmorph[0, ...]).transpose(1, 2, 0)

    ax = fig.add_subplot(6, 7, counter * 7 + 6)
    ax.imshow(warped_image_voxelmorph, vmin=0, vmax=255)
    ax.set_axis_off()
    ax.set_title('Predicted Warped Image\nDiffeoMappingNet (VoxelMorph)', fontsize=16)

    ax = fig.add_subplot(6, 7, counter * 7 + 7)
    vectors = [np.arange(0, s) for s in moving_image.shape[:2]]
    X, Y = np.meshgrid(vectors[0], vectors[1])
    warped_X = X + diffeo_forward_voxelmorph[:, :, 1]
    warped_Y = Y + diffeo_forward_voxelmorph[:, :, 0]
    for i in range(moving_image.shape[0]):
        ax.plot(warped_X[i, :], warped_Y[i, :], color='k')
    for j in range(moving_image.shape[1]):
        ax.plot(warped_X[:, j], warped_Y[:, j], color='k')
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title('Predicted Diffeomorphism\nDiffeoMappingNet (VoxelMorph)', fontsize=16)

    return


if __name__ == '__main__':
    rectangle = random_rectangle(rectangle_size=(32, 32), center=(32, 32))
    star = random_star(center=(32, 32))
    triangle = random_triangle(center=(36, 32))

    # Plot figure.
    plt.rcParams["font.family"] = 'serif'
    fig = plt.figure(figsize=(4*7, 4*6))

    plot_predict_warp(fig, counter=0, moving_image=rectangle, fixed_image=star)
    plot_predict_warp(fig, counter=1, moving_image=star, fixed_image=rectangle)
    plot_predict_warp(fig, counter=2, moving_image=star, fixed_image=triangle)
    plot_predict_warp(fig, counter=3, moving_image=triangle, fixed_image=star)
    plot_predict_warp(fig, counter=4, moving_image=triangle, fixed_image=rectangle)
    plot_predict_warp(fig, counter=5, moving_image=rectangle, fixed_image=triangle)

    fig.tight_layout(pad=2)
    fig.savefig('predict_diffeomorphism.png')

