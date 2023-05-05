import logging
from torch import nn

from .unet_modules import Up, Down, DoubleConv, OutConv


logger = logging.getLogger(__name__)


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
        self.up1 = Up(
            16 * filter_base_count, 8 * filter_base_count // self.factor, self.bilinear, self.use_pooling_indices,
            groups=groups,
        )
        self.up2 = Up(
            8 * filter_base_count, 4 * filter_base_count // self.factor, self.bilinear, self.use_pooling_indices,
            groups=groups,
        )
        self.up3 = Up(
            4 * filter_base_count, 2 * filter_base_count // self.factor, self.bilinear, self.use_pooling_indices,
            groups=groups,
        )
        self.up4 = Up(
            2 * filter_base_count, filter_base_count, self.bilinear, self.use_pooling_indices, 
            groups=groups,
        )
        self.final_dropout = nn.Dropout(p=final_dropout_rate)
        self.outc = OutConv(
            filter_base_count, out_channels,
            groups=groups,
        )

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
