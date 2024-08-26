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
