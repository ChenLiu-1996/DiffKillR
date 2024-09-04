from .base import BaseNetwork
from .blocks import ConvBlock, ResConvBlock, UpConvBlock, ResUpConvBlock
import torch
from torch import nn


class AutoEncoder(BaseNetwork):

    def __init__(self,
                 device: torch.device = torch.device('cpu'),
                 num_filters: int = 16,
                 depth: int = 4,
                 use_residual: bool = True,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 non_linearity: str = 'relu'):
        '''
        An AutoEncoder model.

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

        self.down_list = nn.ModuleList([])
        self.down_conn_list = nn.ModuleList([])
        self.up_list = nn.ModuleList([])
        self.up_conn_list = nn.ModuleList([])

        if self.use_residual:
            conv_block = ResConvBlock
            upconv_block = ResUpConvBlock
        else:
            conv_block = ConvBlock
            upconv_block = UpConvBlock

        for d in range(self.depth):
            self.down_list.append(conv_block(n_f * 2 ** d))
            self.down_conn_list.append(nn.Conv2d(n_f * 2 ** d, n_f * 2 ** (d + 1), 1, 1))
            self.up_conn_list.append(nn.Conv2d(n_f * 2 ** (d + 1), n_f * 2 ** d, 1, 1))
            self.up_list.append(upconv_block(n_f * 2 ** d))

        self.up_list = self.up_list[::-1]
        self.up_conn_list = self.up_conn_list[::-1]

        self.bottleneck = ResConvBlock(n_f * 2 ** self.depth)
        self.out_layer = nn.Conv2d(n_f, out_channels, 1)


    def forward(self, x: torch.Tensor):
        '''
        `interpolate` is used as a drop-in replacement for MaxPool2d.
        '''

        x = self.non_linearity(self.conv1x1(x))

        for d in range(self.depth):
            x = self.down_list[d](x)
            #print(x.shape)
            x = self.non_linearity(self.down_conn_list[d](x))
            #print(x.shape)
            x = nn.functional.interpolate(x,
                                          scale_factor=0.5,
                                          mode='bilinear',
                                          align_corners=False)
            #print(x.shape)

        x = self.bottleneck(x)
        latent = torch.clone(x)
        #print('latent: ', x.shape)

        for d in range(self.depth):
            x = nn.functional.interpolate(x,
                                          scale_factor=2,
                                          mode='bilinear',
                                          align_corners=False)
            #print(x.shape)
            x = self.non_linearity(self.up_conn_list[d](x))
            #print(x.shape)

            x = self.up_list[d](x)
            #print(x.shape)

        output = self.out_layer(x)

        return output, latent
