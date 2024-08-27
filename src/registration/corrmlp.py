import os
import sys
import einops
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir)
from model.base import BaseNetwork


class CorrMLP(BaseNetwork):
    """
    Correlation-aware Coarse-to-fine MLPs for Deformable Medical Image Registration.
    CVPR 2024 (Oral).

    https://arxiv.org/pdf/2406.00123
    https://github.com/MungoMeng/Registration-CorrMLP
    """

    def __init__(self,
                 in_channels: int = 1,
                 enc_channels: int = 16,
                 dec_channels: int = 32,
                 use_checkpoint: bool = True):
        super().__init__()

        self.Encoder = Conv_encoder(in_channels=in_channels,
                                    channel_num=enc_channels,
                                    use_checkpoint=use_checkpoint)
        self.Decoder = MLP_decoder(in_channels=enc_channels,
                                   channel_num=dec_channels,
                                   use_checkpoint=use_checkpoint)

    def forward(self, source, target):

        fixed, moving = source, target
        x_fix = self.Encoder(fixed)
        x_mov = self.Encoder(moving)
        flow_forward = self.Decoder(x_fix, x_mov)
        flow_backward = -flow_forward

        return flow_forward, flow_backward


########################################################
# Encoder/Decoder
########################################################

class Conv_encoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 channel_num: int,
                 use_checkpoint: bool = False):
        super().__init__()

        self.Convblock_1 = Conv_block(in_channels, channel_num, use_checkpoint)
        self.Convblock_2 = Conv_block(channel_num, channel_num*2, use_checkpoint)
        self.Convblock_3 = Conv_block(channel_num*2, channel_num*4, use_checkpoint)
        self.Convblock_4 = Conv_block(channel_num*4, channel_num*8, use_checkpoint)
        self.downsample = nn.AvgPool2d(2, stride=2)

    def forward(self, x_in):

        x_1 = self.Convblock_1(x_in)
        x = self.downsample(x_1)
        x_2 = self.Convblock_2(x)
        x = self.downsample(x_2)
        x_3 = self.Convblock_3(x)
        x = self.downsample(x_3)
        x_4 = self.Convblock_4(x)

        return [x_1, x_2, x_3, x_4]


class MLP_decoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 channel_num: int,
                 use_checkpoint: bool = False):
        super().__init__()

        self.mlp_11 = CMWMLP_block(in_channels, channel_num, use_corr=True, use_checkpoint=use_checkpoint)
        self.mlp_12 = CMWMLP_block(in_channels*2, channel_num*2, use_corr=True, use_checkpoint=use_checkpoint)
        self.mlp_13 = CMWMLP_block(in_channels*4, channel_num*4, use_corr=True, use_checkpoint=use_checkpoint)
        self.mlp_14 = CMWMLP_block(in_channels*8, channel_num*8, use_corr=True, use_checkpoint=use_checkpoint)

        self.mlp_21 = CMWMLP_block(channel_num, channel_num, use_corr=True, use_checkpoint=use_checkpoint)
        self.mlp_22 = CMWMLP_block(channel_num*2, channel_num*2, use_corr=True, use_checkpoint=use_checkpoint)
        self.mlp_23 = CMWMLP_block(channel_num*4, channel_num*4, use_corr=True, use_checkpoint=use_checkpoint)

        self.upsample_1 = PatchExpanding_block(embed_dim=channel_num*2)
        self.upsample_2 = PatchExpanding_block(embed_dim=channel_num*4)
        self.upsample_3 = PatchExpanding_block(embed_dim=channel_num*8)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='bilinear')
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

        self.reghead_1 = RegHead_block(channel_num, use_checkpoint)
        self.reghead_2 = RegHead_block(channel_num*2, use_checkpoint)
        self.reghead_3 = RegHead_block(channel_num*4, use_checkpoint)
        self.reghead_4 = RegHead_block(channel_num*8, use_checkpoint)

    def forward(self, x_fix, x_mov):

        x_fix_1, x_fix_2, x_fix_3, x_fix_4 = x_fix
        x_mov_1, x_mov_2, x_mov_3, x_mov_4 = x_mov

        # Step 1
        x_4 = self.mlp_14(x_fix_4, x_mov_4)
        flow_4 = self.reghead_4(x_4)

        # Step 2
        flow_4_up = self.ResizeTransformer(flow_4)
        x_mov_3 = self.SpatialTransformer(x_mov_3, flow_4_up)

        x = self.mlp_13(x_fix_3, x_mov_3)
        x_3 = self.mlp_23(x, self.upsample_3(x_4))

        x = self.reghead_3(x_3)
        flow_3 = x + flow_4_up

        # Step 3
        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2 = self.SpatialTransformer(x_mov_2, flow_3_up)

        x = self.mlp_12(x_fix_2, x_mov_2)
        x_2 = self.mlp_22(x, self.upsample_2(x_3))

        x = self.reghead_2(x_2)
        flow_2 = x + flow_3_up

        # Step 4
        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1 = self.SpatialTransformer(x_mov_1, flow_2_up)

        x = self.mlp_11(x_fix_1, x_mov_1)
        x_1 = self.mlp_21(x, self.upsample_1(x_2))

        x = self.reghead_1(x_1)
        flow_1 = x + flow_2_up

        return flow_1


########################################################
# Blocks
########################################################

class SpatialTransformer_block(nn.Module):

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)

        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1,0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class ResizeTransformer_block(nn.Module):

    def __init__(self, resize_factor, mode='bilinear'):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        return x


class Conv_block(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels:int ,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.Conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.norm_1 = nn.InstanceNorm2d(out_channels)

        self.Conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.norm_2 = nn.InstanceNorm2d(out_channels)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def Conv_forward(self, x_in):

        x = self.Conv_1(x_in)
        x = self.LeakyReLU(x)
        x = self.norm_1(x)

        x = self.Conv_2(x)
        x = self.LeakyReLU(x)
        x_out = self.norm_2(x)

        return x_out

    def forward(self, x_in):

        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.Conv_forward, x_in)
        else:
            x_out = self.Conv_forward(x_in)

        return x_out


class RegHead_block(nn.Module):

    def __init__(self,
                 in_channels: int,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.reg_head = nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding='same')
        self.reg_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.reg_head.weight.shape))
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))

    def forward(self, x_in):

        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.reg_head, x_in)
        else:
            x_out = self.reg_head(x_in)

        return x_out


class PatchExpanding_block(nn.Module):

    def __init__(self, embed_dim: int):
        super().__init__()

        self.up_conv = nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(embed_dim//2)

    def forward(self, x_in):

        x = self.up_conv(x_in)
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        x_out = einops.rearrange(x, 'b h w c -> b c h w')

        return x_out


class CMWMLP_block(nn.Module):  #input shape: [n, c, h, w]
    """Correlation-aware multi-window (CMW) MLP block."""

    def __init__(self, in_channels, num_channels, use_corr=True, use_checkpoint=False):
        super().__init__()
        self.use_corr = use_corr

        if use_corr:
            self.Corr = Correlation(max_disp=1, use_checkpoint=use_checkpoint)
            self.Conv = nn.Conv2d(in_channels*2+9, num_channels, kernel_size=3, stride=1, padding='same')
        else:
            self.Conv = nn.Conv2d(in_channels*2, num_channels, kernel_size=3, stride=1, padding='same')

        self.mlpLayer = MultiWinMlpLayer(num_channels, use_checkpoint=use_checkpoint)
        self.channel_attention_block = RCAB(num_channels, use_checkpoint=use_checkpoint)

    def forward(self, x_1, x_2):

        if self.use_corr:
            x_corr = self.Corr(x_1, x_2)
            x = torch.cat([x_1, x_corr, x_2], dim=1)
            x = self.Conv(x)
        else:
            x = torch.cat([x_1, x_2], dim=1)
            x = self.Conv(x)

        shortcut = x
        x = x.permute(0, 2, 3, 1)  # [n, h, w, c]
        x = self.mlpLayer(x)
        x = self.channel_attention_block(x)
        x = x.permute(0, 3, 1, 2)  # [n, c, h, w]

        x_out = x + shortcut
        return x_out


class MultiWinMlpLayer(nn.Module):   #input shape: [n, h, w, c]
    """The multi-window gated MLP block."""

    def __init__(self, num_channels, use_bias=True, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.LayerNorm = nn.LayerNorm(num_channels)
        self.WinGmlpLayer_1 = WinGmlpLayer(win_size=[3,3,3], num_channels=num_channels, use_bias=use_bias)
        self.WinGmlpLayer_2 = WinGmlpLayer(win_size=[5,5,5], num_channels=num_channels, use_bias=use_bias)
        self.WinGmlpLayer_3 = WinGmlpLayer(win_size=[7,7,7], num_channels=num_channels, use_bias=use_bias)

        self.reweight = MLP(num_channels, num_channels//4, num_channels * 3)
        self.out_project = nn.Linear(num_channels, num_channels, bias=use_bias)

    def forward_run(self, x_in):

        n, h, w, c = x_in.shape
        x = self.LayerNorm(x_in)

        #Window gMLP
        x_1 = self.WinGmlpLayer_1(x)
        x_2 = self.WinGmlpLayer_2(x)
        x_3 = self.WinGmlpLayer_3(x)

        a = (x_1 + x_2 + x_3).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(n, c, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = x_1*a[0] + x_2*a[1] + x_3*a[2]
        x = self.out_project(x)

        x_out = x + x_in
        return x_out

    def forward(self, x_in):

        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.forward_run, x_in)
        else:
            x_out = self.forward_run(x_in)
        return x_out


class WinGmlpLayer(nn.Module):  #input shape: [n, h, w, c]

    def __init__(self, win_size, num_channels, factor=2, use_bias=True):
        super().__init__()

        self.fh = win_size[0]
        self.fw = win_size[1]

        self.LayerNorm = nn.LayerNorm(num_channels)
        self.in_project = nn.Linear(num_channels, num_channels*factor, use_bias)   #c->c*factor
        self.gelu = nn.GELU()
        self.SpatialGatingUnit = SpatialGatingUnit(num_channels*factor, n=self.fh*self.fw)   #c*factor->c*factor//2
        self.out_project = nn.Linear(num_channels*factor//2, num_channels, use_bias)   #c*factor//2->c

    def forward(self, x):

        _, h, w, _ = x.shape

        # padding
        pad_t = pad_l = 0
        pad_r = (self.fw - w % self.fw) % self.fw
        pad_b = (self.fh - h % self.fh) % self.fh
        x = nnf.pad(x, (0, 0, pad_t, pad_b, pad_l, pad_r))

        gh, gw = x.shape[1] // self.fh, x.shape[2] // self.fw
        x = split_images(x, patch_size=(self.fh, self.fw))  #n (gh gw) (fh fw) c

        # gMLP: Local (block) mixing part, provides local block communication.
        shortcut = x
        x = self.LayerNorm(x)
        x = self.in_project(x)
        x = self.gelu(x)
        x = self.SpatialGatingUnit(x)
        x = self.out_project(x)
        x = x + shortcut

        x = unsplit_images(x, grid_size=(gh, gw), patch_size=(self.fh, self.fw))
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        return x


class SpatialGatingUnit(nn.Module):  #input shape: n (gh gw gd) (fh fw fd) c

    def __init__(self, c, n, use_bias=True):
        super().__init__()

        self.Dense_0 = nn.Linear(n, n, use_bias)
        self.LayerNorm = nn.LayerNorm(c//2)

    def forward(self, x):

        c = x.size(-1)
        c = c // 2
        u, v  = torch.split(x, c, dim=-1)

        v = self.LayerNorm(v)
        v = v.permute(0, 1, 3, 2)  #n, (gh gw gd), c/2, (fh fw fd)
        v = self.Dense_0(v)
        v = v.permute(0, 1, 3, 2)  #n (gh gw gd) (fh fw fd) c/2

        return u*(v + 1.0)


class RCAB(nn.Module):  #input shape: n, h, w, d, c
    """Residual channel attention block. Contains LN,Conv,lRelu,Conv,SELayer."""

    def __init__(self, num_channels, reduction=4, lrelu_slope=0.2, use_bias=True, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.LayerNorm = nn.LayerNorm(num_channels)
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, bias=use_bias, padding='same')
        self.leaky_relu = nn.LeakyReLU(negative_slope=lrelu_slope)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, bias=use_bias, padding='same')
        self.channel_attention = CALayer(num_channels=num_channels, reduction=reduction)

    def forward_run(self, x):

        shortcut = x
        x = self.LayerNorm(x)

        x = x.permute(0, 3, 1, 2)  # [n, c, h, w]
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)  # [n, h, w, c]

        x = self.channel_attention(x)
        x_out = x + shortcut

        return x_out

    def forward(self, x):

        if self.use_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(self.forward_run, x)
        else:
            x = self.forward_run(x)
        return x


class CALayer(nn.Module):  #input shape: n, h, w, c
    """Squeeze-and-excitation block for channel attention."""

    def __init__(self, num_channels, reduction=4, use_bias=True):
        super().__init__()

        self.Conv_0 = nn.Conv2d(num_channels, num_channels//reduction, kernel_size=1, stride=1, bias=use_bias)
        self.relu = nn.ReLU()
        self.Conv_1 = nn.Conv2d(num_channels//reduction, num_channels, kernel_size=1, stride=1, bias=use_bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_in):

        x = x_in.permute(0, 3, 1, 2)  # [n, c, h, w]
        x = torch.mean(x, dim=(2,3), keepdim=True)
        x = self.Conv_0(x)
        x = self.relu(x)
        x = self.Conv_1(x)
        w = self.sigmoid(x)
        w = w.permute(0, 2, 3, 1)  # [n, h, w, c]

        x_out = x_in*w
        return x_out


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Correlation(nn.Module):
    def __init__(self, max_disp=1, kernel_size=1, stride=1, use_checkpoint=False):
        assert kernel_size == 1, "kernel_size other than 1 is not implemented"
        assert stride == 1, "stride other than 1 is not implemented"
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.max_disp = max_disp
        self.padlayer = nn.ConstantPad2d(max_disp, 0)

    def forward_run(self, x_1, x_2):

        x_2 = self.padlayer(x_2)
        offsetx, offsety = torch.meshgrid([torch.arange(0, 2 * self.max_disp + 1),
                                           torch.arange(0, 2 * self.max_disp + 1)], indexing='ij')

        w, h = x_1.shape[2], x_1.shape[3]
        x_out = torch.cat([torch.mean(x_1 * x_2[:, :, dx:dx+w, dy:dy+h], 1, keepdim=True)
                           for dx, dy in zip(offsetx.reshape(-1), offsety.reshape(-1))], 1)
        return x_out

    def forward(self, x_1, x_2):

        if self.use_checkpoint and x_1.requires_grad and x_2.requires_grad:
            x = checkpoint.checkpoint(self.forward_run, x_1, x_2)
        else:
            x = self.forward_run(x_1, x_2)
        return x


########################################################
# Functions
########################################################

def split_images(x, patch_size):  #n, h, w, c
    """Image to patches."""

    batch, height, width, channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]

    x = einops.rearrange(
        x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
        gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x


def unsplit_images(x, grid_size, patch_size):
    """patches to images."""

    x = einops.rearrange(
        x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
        gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
    return x


if __name__ == '__main__':
    import os
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

    DiffeoMappingNet = CorrMLP(
        in_channels=3,
    )
    DiffeoMappingNet = DiffeoMappingNet.to(device)
    DiffeoMappingNet.train()

    warper = SpatialTransformer(size=moving_image.shape[:2])
    warper = warper.to(device)

    optimizer = torch.optim.AdamW(DiffeoMappingNet.parameters(), lr=1e-4)
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
    diffeo_forward_corrmlp, _ = DiffeoMappingNet(source=moving_image_torch, target=fixed_image_torch)
    diffeo_forward_corrmlp = diffeo_forward_corrmlp.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)
    warped_image = warper(torch.from_numpy(moving_image.transpose(2, 0, 1)[None, ...]).float(),
                               flow=torch.from_numpy(diffeo_forward_corrmlp.transpose(2, 0, 1)[None, ...]))
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
    ax.set_title('Predicted Warped Image\nDiffeoMappingNet (CorrMLP)', fontsize=16)

    ax = fig.add_subplot(1, 4, 4)
    vectors = [np.arange(0, s) for s in moving_image.shape[:2]]
    X, Y = np.meshgrid(vectors[0], vectors[1])
    warped_X = X + diffeo_forward_corrmlp[:, :, 1]
    warped_Y = Y + diffeo_forward_corrmlp[:, :, 0]
    for i in range(moving_image.shape[0]):
        ax.plot(warped_X[i, :], warped_Y[i, :], color='k')
    for j in range(moving_image.shape[1]):
        ax.plot(warped_X[:, j], warped_Y[:, j], color='k')
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title('Predicted Diffeomorphism\nDiffeoMappingNet (CorrMLP)', fontsize=16)

    fig.tight_layout(pad=2)
    os.makedirs('./test_output/', exist_ok=True)
    fig.savefig('./test_output/test_corrmlp.png')
