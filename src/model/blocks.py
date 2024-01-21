from torch import nn


class ConvBlock(nn.Module):

    def __init__(self, num_filters):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_filters,
                      num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.InstanceNorm2d(num_filters),
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
                      bias=True),
            nn.InstanceNorm2d(num_filters),
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
                      bias=True),
            nn.InstanceNorm2d(num_filters),
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
                      bias=True),
            nn.InstanceNorm2d(num_filters),
            nn.Conv2d(num_filters,
                      num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True))

    def forward(self, x):
        return self.upconv(x) + x
