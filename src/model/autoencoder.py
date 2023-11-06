from .base import BaseNetwork
import torch
from torch import nn


class AutoEncoder(BaseNetwork):

    def __init__(self,
                 device: torch.device = torch.device('cpu'),
                 num_filters: int = 16,
                 depth: int = 4,
                 use_residual: bool = False,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 non_linearity: str = 'relu'):
        '''
        A Residual AutoEncoder model with ODE.

        Parameters
        ----------
        device: torch.device
        num_filters : int
            Number of convolutional filters.
        depth : int
            Depth of the network.
        use_residual : bool
            Whether to use residual connections.
        in_channels: int
            Number of input image channels.
        out_channels: int
            Number of output image channels.
        non_linearity : string
            One of 'relu' and 'softplus'
        '''
        super().__init__()

        self.device = device
        self.in_channels = in_channels
        self.depth = depth
        self.use_residual = use_residual
        self.non_linearity_str = non_linearity
        if self.non_linearity_str == 'relu':
            self.non_linearity = nn.ReLU(inplace=True)
        elif self.non_linearity_str == 'softplus':
            self.non_linearity = nn.Softplus()

        n_f = num_filters  # shorthand

        self.conv1x1 = nn.Conv2d(in_channels, n_f, 1, 1)

        self.down_list = []
        self.down_conn_list = []
        self.up_list = []
        self.up_conn_list = []

        if self.use_residual:
            conv_block = ResConvBlock
        else:
            conv_block = ConvBlock

        for d in range(self.depth):
            self.down_list.append(conv_block(n_f * 2 ** d))
            self.down_conn_list.append(nn.Conv2d(n_f * 2 ** d, n_f * 2 ** (d + 1), 1, 1))
            self.up_conn_list.append(nn.Conv2d(n_f * 2 ** (d + 1), n_f * 2 ** d, 1, 1))
            self.up_list.append(conv_block(n_f * 2 ** d))

        self.up_list = self.up_list[::-1]
        self.up_conn_list = self.up_conn_list[::-1]

        self.bottleneck = ResConvBlock(n_f * 2 ** self.depth)
        self.out_layer = nn.Conv2d(n_f, out_channels, 1)


    def forward(self, x: torch.Tensor):
        '''
        `interpolate` is used as a drop-in replacement for MaxPool2d.
        '''

        assert x.shape[0] == 1

        x = self.non_linearity(self.conv1x1(x))

        for d in range(self.depth):
            x = self.down_list[d](x)
            x = self.non_linearity(self.down_conn_list[d](x))
            x = nn.functional.interpolate(x,
                                          scale_factor=0.5,
                                          mode='bilinear',
                                          align_corners=False)

        x = self.bottleneck(x)

        for d in range(self.depth):
            x = nn.functional.interpolate(x,
                                          scale_factor=2,
                                          mode='bilinear',
                                          align_corners=False)
            x = self.non_linearity(self.up_conn_list[d](x))
            x = self.up_list[d](x)

        output = self.out_layer(x)

        return output


class ConvBlock(nn.Module):

    def __init__(self, num_filters):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_filters,
                      num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.InstanceNorm2d(num_filters),
            nn.Conv2d(num_filters,
                      num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True))

    def forward(self, x):
        return self.conv(x)


class ResConvBlock(nn.Module):

    def __init__(self, num_filters):
        super(ResConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_filters,
                      num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.InstanceNorm2d(num_filters),
            nn.Conv2d(num_filters,
                      num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True))

    def forward(self, x):
        return self.conv(x) + x


class UpConvBlock(nn.Module):

    def __init__(self, num_filters):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.Sequential(
            nn.Conv2d(num_filters,
                      num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.InstanceNorm2d(num_filters),
            nn.Conv2d(num_filters,
                      num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True))

    def forward(self, x):
        return self.upconv(x)


class ResUpConvBlock(nn.Module):

    def __init__(self, num_filters):
        super(ResUpConvBlock, self).__init__()
        self.upconv = nn.Sequential(
            nn.Conv2d(num_filters,
                      num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.InstanceNorm2d(num_filters),
            nn.Conv2d(num_filters,
                      num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True))

    def forward(self, x):
        return self.upconv(x) + x
