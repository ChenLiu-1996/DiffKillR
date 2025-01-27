#
# https://github.com/voxelmorph/voxelmorph
#
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as nnf

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir)
from model.base import BaseNetwork


class VxmDense(BaseNetwork):
    """
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration.
    CVPR 2018, IEEE TMI.

    https://arxiv.org/pdf/1809.05231
    https://github.com/voxelmorph/voxelmorph

    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # # configure transformer
        # self.transformer = SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        return pos_flow, neg_flow


def default_unet_features(smallest_dim: int):
    if smallest_dim < 16:
        nb_features = [
            [16, 32],     # encoder
            [32, 16, 16]  # decoder
        ]
    elif smallest_dim < 32:
        nb_features = [
            [16, 32, 32],         # encoder
            [32, 32, 32, 16, 16]  # decoder
        ]
    else:
        nb_features = [
            [16, 32, 32, 32],             # encoder
            [32, 32, 32, 32, 32, 16, 16]  # decoder
        ]
    return nb_features

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
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

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features(np.min(inshape))

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


if __name__ == '__main__':
    import numpy as np
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    from spatial_transformer import SpatialTransformer
    from registration_utils import random_rectangle, random_star, radially_color_mask_with_colormap

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    image_size = (64, 64)
    rectangle = random_rectangle(image_size=image_size, rectangle_size=(32, 32), center=(32, 32))
    star = random_star(image_size=image_size, center=(32, 32))

    moving_image, fixed_image = rectangle, star
    moving_image = radially_color_mask_with_colormap(moving_image)
    fixed_image = radially_color_mask_with_colormap(fixed_image)

    # NOTE: `int_steps=0` gives VM (non-diffeomorphic) and `int_steps=7` gives VM-Diff (diffeomorphic)
    DiffeoMappingNet = VxmDense(
        inshape=image_size,
        src_feats=3,
        trg_feats=3,
        int_steps=0,
        bidir=True)
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
    diffeo_forward_vm, _ = DiffeoMappingNet(source=moving_image_torch, target=fixed_image_torch)
    diffeo_forward_vm = diffeo_forward_vm.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)
    warped_image = warper(torch.from_numpy(moving_image.transpose(2, 0, 1)[None, ...]).float(),
                               flow=torch.from_numpy(diffeo_forward_vm.transpose(2, 0, 1)[None, ...]))
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
    ax.set_title('Predicted Warped Image\nDiffeoMappingNet (VM)', fontsize=16)

    ax = fig.add_subplot(1, 4, 4)
    vectors = [np.arange(0, s) for s in moving_image.shape[:2]]
    X, Y = np.meshgrid(vectors[0], vectors[1])
    warped_X = X + diffeo_forward_vm[:, :, 1]
    warped_Y = Y + diffeo_forward_vm[:, :, 0]
    for i in range(moving_image.shape[0]):
        ax.plot(warped_X[i, :], warped_Y[i, :], color='k')
    for j in range(moving_image.shape[1]):
        ax.plot(warped_X[:, j], warped_Y[:, j], color='k')
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title('Predicted Diffeomorphism\nDiffeoMappingNet (VM)', fontsize=16)

    fig.tight_layout(pad=2)
    os.makedirs('./test_output/', exist_ok=True)
    fig.savefig('./test_output/test_voxelmorph.png')
