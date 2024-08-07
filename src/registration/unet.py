#
# https://github.com/voxelmorph/voxelmorph
#
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import torch
import torch.nn as nn
import torch.nn.functional as nnf

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
