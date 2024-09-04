#
# https://github.com/voxelmorph/voxelmorph
#
import os
import sys
import torch

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir)
from model.unet import UNet as base_unet
from model.base import BaseNetwork


class UNet(BaseNetwork):
    """
    UNet for Registration.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.unet_model = base_unet(*args, **kwargs)

    def forward(self, source, target):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        warp_predicted = self.unet_model(x)

        warp_forwrad = warp_predicted[:, :2, ...]
        warp_reverse = warp_predicted[:, 2:, ...]
        return warp_forwrad, warp_reverse


if __name__ == '__main__':
    import numpy as np
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    from spatial_transformer import SpatialTransformer
    from registration_utils import random_rectangle, random_star, radially_color_mask_with_colormap

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    rectangle = random_rectangle(rectangle_size=(32, 32), center=(32, 32))
    star = random_star(center=(32, 32))

    moving_image, fixed_image = rectangle, star
    moving_image = radially_color_mask_with_colormap(moving_image)
    fixed_image = radially_color_mask_with_colormap(fixed_image)

    DiffeoMappingNet = UNet(
        num_filters=32,
        in_channels=6,
        out_channels=4)
    DiffeoMappingNet = DiffeoMappingNet.to(device)
    DiffeoMappingNet.train()

    warper = SpatialTransformer(size=moving_image.shape[:2])
    warper = warper.to(device)

    optimizer = torch.optim.AdamW(DiffeoMappingNet.parameters(), lr=1e-3)
    mse_loss = torch.nn.MSELoss()

    moving_image_torch = torch.from_numpy((moving_image).transpose(2, 0, 1)[None, ...]).float()
    fixed_image_torch = torch.from_numpy((fixed_image).transpose(2, 0, 1)[None, ...]).float()

    for _ in tqdm(range(120)):
        __diffeo_forward, __diffeo_backward = DiffeoMappingNet(source=moving_image_torch, target=fixed_image_torch)
        __image_warped_forward = warper(moving_image_torch, flow=__diffeo_forward)
        __image_warped_backward = warper(fixed_image_torch, flow=__diffeo_backward)

        loss_forward = mse_loss(fixed_image_torch, __image_warped_forward)
        loss_cyclic = mse_loss(moving_image_torch, __image_warped_backward)
        loss = loss_forward + loss_cyclic
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    DiffeoMappingNet.eval()
    diffeo_forward_unet, _ = DiffeoMappingNet(source=moving_image_torch, target=fixed_image_torch)
    diffeo_forward_unet = diffeo_forward_unet.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)
    warped_image = warper(torch.from_numpy(moving_image.transpose(2, 0, 1)[None, ...]).float(),
                               flow=torch.from_numpy(diffeo_forward_unet.transpose(2, 0, 1)[None, ...]))
    warped_image = np.uint8(warped_image[0, ...]).transpose(1, 2, 0)

    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(moving_image, vmin=0, vmax=255)
    ax.set_axis_off()
    ax.set_title('Original Image', fontsize=16)

    ax = fig.add_subplot(1, 4, 2)
    ax.imshow(fixed_image, vmin=0, vmax=255)
    ax.set_axis_off()
    ax.set_title('Fixed Image', fontsize=16)

    ax = fig.add_subplot(1, 4, 3)
    ax.imshow(warped_image, vmin=0, vmax=255)
    ax.set_axis_off()
    ax.set_title('Predicted Warped Image\nDiffeoMappingNet (UNet)', fontsize=16)

    ax = fig.add_subplot(1, 4, 4)
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

    fig.tight_layout(pad=2)
    os.makedirs('./test_output/', exist_ok=True)
    fig.savefig('./test_output/test_unet.png')
