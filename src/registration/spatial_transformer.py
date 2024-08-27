import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer

    https://github.com/voxelmorph/voxelmorph/blob/ca3d47a2c254aae9a0c0e1b30c24c324c211ebc8/voxelmorph/torch/layers.py
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


if __name__ == '__main__':
    import os
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
    from dipy.align.metrics import CCMetric

    fixed_image = cv2.cvtColor(
        cv2.imread('../../data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00_augmented_patch_32x32/partial_stretch/image/EpithelialCell_H6852_W13114_original.png'),
        cv2.COLOR_BGR2RGB)
    moving_image = cv2.cvtColor(
        cv2.imread('../../data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00_augmented_patch_32x32/partial_stretch/image/EpithelialCell_H3018_W10206_original.png'),
        cv2.COLOR_BGR2RGB)

    # Use classic registration to generate the warping field.
    # This is just for unit test / sanity check.
    # In practice, we will use a neural network to learn this warping field.
    metric = CCMetric(2, sigma_diff=2, radius=2)
    sdr = SymmetricDiffeomorphicRegistration(metric)
    mapping = sdr.optimize(fixed_image[..., 0], moving_image[..., 0])

    # The warp operation performed by classic registration
    warped_dipy = np.zeros_like(fixed_image)
    warped_dipy[..., 0] = mapping.transform(moving_image[..., 0])
    warped_dipy[..., 1] = mapping.transform(moving_image[..., 1])
    warped_dipy[..., 2] = mapping.transform(moving_image[..., 2])

    # The warp operation performed by Sptial Transformer.
    spatial_trans = SpatialTransformer(size=(32, 32))
    warped_st = spatial_trans(torch.from_numpy(moving_image.transpose(2, 0, 1)[None, ...]).float(),
                              flow=torch.from_numpy(mapping.backward.transpose(2, 0, 1)[None, ...]))
    warped_st = np.uint8(warped_st[0, ...]).transpose(1, 2, 0)

    # Plotting.
    plt.rcParams['font.family'] = 'serif'
    fig = plt.figure(figsize=(12, 9))

    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(fixed_image)
    ax.set_axis_off()
    ax.set_title('Fixed image')

    ax = fig.add_subplot(2, 3, 4)
    ax.imshow(moving_image)
    ax.set_axis_off()
    ax.set_title('Moving image')

    ax = fig.add_subplot(2, 3, 2)
    vectors = [np.arange(0, s) for s in (32, 32)]
    X, Y = np.meshgrid(vectors[0], vectors[1])
    warped_X = X + mapping.backward[:, :, 0]
    warped_Y = Y + mapping.backward[:, :, 1]
    for i in range(32):
        ax.plot(warped_X[i, :], warped_Y[i, :], color='k')
    for j in range(32):
        ax.plot(warped_X[:, j], warped_Y[:, j], color='k')
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title('Forward warp')

    ax = fig.add_subplot(2, 3, 5)
    vectors = [np.arange(0, s) for s in (32, 32)]
    X, Y = np.meshgrid(vectors[0], vectors[1])
    warped_X = X + mapping.forward[:, :, 0]
    warped_Y = Y + mapping.forward[:, :, 1]
    for i in range(32):
        ax.plot(warped_X[i, :], warped_Y[i, :], color='k')
    for j in range(32):
        ax.plot(warped_X[:, j], warped_Y[:, j], color='k')
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title('Inverse warp')

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(warped_dipy)
    ax.set_axis_off()
    ax.set_title('Warp operation by\nClassic registration')

    ax = fig.add_subplot(2, 3, 6)
    ax.imshow(warped_st)
    ax.set_axis_off()
    ax.set_title('Warp operation by\nDifferentiable Neural Network')

    fig.tight_layout(pad=2.0)
    os.makedirs('./test_output/', exist_ok=True)
    fig.savefig('./test_output/test_spatial_transformer.png')
