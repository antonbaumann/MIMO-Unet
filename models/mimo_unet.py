import logging
import torch
from torch import nn
import torch.utils.data

from unet_modules import Up, Down, DoubleConv, OutConv

logger = logging.getLogger(__name__)


class MimoUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_subnetworks: int = 1,
        bilinear: bool = True,
        filter_base_count: int = 30,
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
        super(MimoUNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear
        self.use_pooling_indices = use_pooling_indices
        self.filter_base_count = filter_base_count

        self.in_convs = nn.ModuleList()
        for i in range(num_subnetworks):
            self.in_convs.append(DoubleConv(in_channels, filter_base_count))

        self.down1s = nn.ModuleList()
        for i in range(num_subnetworks):
            self.down1s.append(Down(
                in_channels=filter_base_count, 
                out_channels=2 * filter_base_count, 
                use_pooling_indices=self.use_pooling_indices,
            ))

        self.down2 = Down(
            in_channels=2 * filter_base_count * num_subnetworks, 
            out_channels=4 * filter_base_count * num_subnetworks,
            use_pooling_indices=self.use_pooling_indices,
        )
        self.down3 = Down(
            in_channels=4 * filter_base_count * num_subnetworks, 
            out_channels=8 * filter_base_count * num_subnetworks,
            use_pooling_indices=self.use_pooling_indices,
        )
        self.factor = 2 if (self.bilinear or self.use_pooling_indices) else 1
        self.down4 = Down(
            in_channels=8 * filter_base_count * num_subnetworks, 
            out_channels=16 * filter_base_count * num_subnetworks // self.factor, 
            use_pooling_indices=self.use_pooling_indices,
        )
        self.center_dropout = nn.Dropout(p=center_dropout_rate)
        self.up1 = Up(
            in_channels=16 * filter_base_count * num_subnetworks, 
            out_channels=8 * filter_base_count * num_subnetworks // self.factor, 
            bilinear=self.bilinear, 
            use_pooling_indices=self.use_pooling_indices,
        )
        self.up2 = Up(
            in_channels=8 * filter_base_count * num_subnetworks, 
            out_channels=4 * filter_base_count * num_subnetworks // self.factor, 
            bilinear=self.bilinear, 
            use_pooling_indices=self.use_pooling_indices,
        )
        self.up3 = Up(
            in_channels=4 * filter_base_count * num_subnetworks, 
            out_channels=2 * filter_base_count * num_subnetworks // self.factor, 
            bilinear=self.bilinear, 
            use_pooling_indices=self.use_pooling_indices,
        )

        self.up4s = nn.ModuleList()
        for i in range(num_subnetworks):
            self.up4s.append(Up(
                in_channels=2 * filter_base_count * num_subnetworks // self.factor + filter_base_count, 
                out_channels=filter_base_count, 
                bilinear=self.bilinear, 
                use_pooling_indices=self.use_pooling_indices, 
            ))

        self.final_dropouts = nn.ModuleList()
        for i in range(num_subnetworks):
            self.final_dropouts.append(
                nn.Dropout(p=final_dropout_rate)
            )

        self.outcs = nn.ModuleList()
        for i in range(num_subnetworks):
            self.outcs.append(OutConv(
                in_channels=filter_base_count, 
                out_channels=out_channels,
            ))

    def forward(self, x):
        """
        Args:
            x: [B, S, C_in, H, W]
            - B: batch size
            - S: number of subnetworks
            - C_in: number of input channels per subnetwork
            - H, W: image dimensions
        """

        _, S, _, _, _ = x.shape

        x1s = []
        x2s = []
        ind2s = []

        for i in range(S):
            x1 = self.in_convs[i](x[:, i])
            x2, ind2 = self.down1s[i](x1)

            x1s.append(x1)
            x2s.append(x2)
            ind2s.append(ind2)
        
        # concatenate along channel dimension
        x2_concat = torch.cat(x2s, axis=1)

        x3, ind3 = self.down2(x2_concat)
        x4, ind4 = self.down3(x3)
        x5, ind5 = self.down4(x4)
        x_drop = self.center_dropout(x5)
        x = self.up1(x_drop, x4, ind5)
        x = self.up2(x, x3, ind4)
        x = self.up3(x, x2_concat, ind3)

        logits = []
        for i in range(S):
            x_i = self.up4s[i](x, x1s[i], ind2s[i])
            x_i = self.final_dropouts[i](x_i)
            logits.append(self.outcs[i](x_i))
        return torch.stack(logits, axis=1)

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
