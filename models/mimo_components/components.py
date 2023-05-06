from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, groups: int = 1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode="reflect", groups=groups),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect", groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, use_pooling_indices: bool = False):
        super().__init__()
        self.use_pooling_indices = use_pooling_indices
        self.maxpool = nn.MaxPool2d(2, return_indices=self.use_pooling_indices)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y = self.maxpool(x)
        if self.use_pooling_indices:
            y, indices = y
        y = self.conv(y)
        result = (y, indices) if self.use_pooling_indices else (y, None)
        return result


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, use_pooling_indices: bool = False, groups: int = 1):
        super().__init__()
        assert int(bilinear) + int(use_pooling_indices) <= 1, "Do not specify use_pooling_indices and bilinear together!"
        self.use_pooling_indices = use_pooling_indices

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, groups=groups)
        elif self.use_pooling_indices:
            self.up = nn.MaxUnpool2d(2, padding=0)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, groups=groups)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2, groups=groups,
            )
            self.conv = DoubleConv(in_channels, out_channels, groups=groups)

    def forward(self, x1, x2, pooling_indices: Optional = None):
        if self.use_pooling_indices:
            x1 = self.up(x1, indices=pooling_indices)
        else:
            x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups: int = 1):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups)

    def forward(self, x):
        return self.conv(x)
