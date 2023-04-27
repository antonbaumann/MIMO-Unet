from typing import Optional
import logging
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect"),
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

    def __init__(self, in_channels, out_channels, bilinear=True, use_pooling_indices: bool = False):
        super().__init__()
        assert int(bilinear) + int(use_pooling_indices) <= 1, "Do not specify use_pooling_indices and bilinear together!"
        self.use_pooling_indices = use_pooling_indices

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        elif self.use_pooling_indices:
            self.up = nn.MaxUnpool2d(2, padding=0)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

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


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 1,
        bilinear: bool = True,
        filter_base_count: int = 60,
        center_dropout_rate: float = 0.0,
        final_dropout_rate: float = 0.0,
        use_pooling_indices: bool = False,
        **kwargs, ## ignore additional arguments
    ):
        logger.info(
            "Creating UNet with arguments: n_channels=%d, n_classes=%d, bilinear=%s, filter_base_count=%d, "
            "center_dropout_rate=%f, final_dropout_rate=%f, use_pooling_indices=%s",
            in_channels,
            out_channels,
            bilinear,
            filter_base_count,
            center_dropout_rate,
            final_dropout_rate,
            use_pooling_indices,
        )
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear
        self.use_pooling_indices = use_pooling_indices
        self.filter_base_count = filter_base_count

        self.inc = DoubleConv(in_channels, filter_base_count)
        self.down1 = Down(filter_base_count, 2 * filter_base_count, self.use_pooling_indices)
        self.down2 = Down(2 * filter_base_count, 4 * filter_base_count, self.use_pooling_indices)
        self.down3 = Down(4 * filter_base_count, 8 * filter_base_count, self.use_pooling_indices)
        self.factor = 2 if (self.bilinear or self.use_pooling_indices) else 1
        self.down4 = Down(8 * filter_base_count, 16 * filter_base_count // self.factor, self.use_pooling_indices)
        self.center_dropout = nn.Dropout(p=center_dropout_rate)
        self.up1 = Up(16 * filter_base_count, 8 * filter_base_count // self.factor, self.bilinear, self.use_pooling_indices)
        self.up2 = Up(8 * filter_base_count, 4 * filter_base_count // self.factor, self.bilinear, self.use_pooling_indices)
        self.up3 = Up(4 * filter_base_count, 2 * filter_base_count // self.factor, self.bilinear, self.use_pooling_indices)
        self.up4 = Up(2 * filter_base_count, filter_base_count, self.bilinear, self.use_pooling_indices)
        self.final_dropout = nn.Dropout(p=final_dropout_rate)
        self.outc = OutConv(filter_base_count, out_channels, groups=groups)

    def forward(self, x):
        x1 = self.inc(x)
        x2, ind2 = self.down1(x1)
        x3, ind3 = self.down2(x2)
        x4, ind4 = self.down3(x3)
        x5, ind5 = self.down4(x4)
        x_drop = self.center_dropout(x5)
        x = self.up1(x_drop, x4, ind5)
        x = self.up2(x, x3, ind4)
        x = self.up3(x, x2, ind3)
        x = self.up4(x, x1, ind2)
        x = self.final_dropout(x)
        logits = self.outc(x)
        return logits

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("UNet")
        parser.add_argument("--no_unet_bilinear", dest="bilinear", action="store_false")
        parser.add_argument("--unet_bilinear", dest="bilinear", action="store_true")

        parser.add_argument(
            "--center_dropout_rate",
            type=float,
            help="Set dropout at the center of the UNet",
            default=0.0,
        )
        parser.add_argument(
            "--final_dropout_rate",
            type=float,
            help="Set dropout before the last conv layer UNet",
            default=0.0,
        )
        parser.add_argument(
            "--filter_base_count",
            type=int,
            help="Set the base number of filters.",
            default=64,
        )
        parser.add_argument(
            "--use_pooling_indices",
            default=False,
            help="Whether to use the pooling index at unpooling or not.",
            action="store_true",
        )
        return parent_parser
