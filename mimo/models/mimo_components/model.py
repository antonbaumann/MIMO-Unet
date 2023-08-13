from typing import List, Tuple
import logging
import torch
from torch import nn
import torch.utils.data

from .components import Up, Down, DoubleConv, OutConv

logger = logging.getLogger(__name__)


def create_module_list(module: nn.Module, num_subnetworks: int, **kwargs):
    """
    Utility function to create a list of identical modules.
    
    Args:
        module: The PyTorch module to be repeated.
        num_subnetworks: Number of times the module needs to be repeated.
        **kwargs: Arguments to be passed to the module during initialization.
    
    Returns:
        A module list containing the repeated modules.
    """
    return nn.ModuleList([module(**kwargs) for _ in range(num_subnetworks)])

class MimoUNet(nn.Module):
    """
    Multiple Input Multiple Output (MIMO) UNet architecture.
    Consists of an encoder, a core, and a decoder.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_subnetworks: int,
        filter_base_count: int = 30,
        center_dropout_rate: float = 0.0,
        final_dropout_rate: float = 0.0,
        encoder_dropout_rate: float = 0.0,
        core_dropout_rate: float = 0.0,
        decoder_dropout_rate: float = 0.0,
        bilinear: bool = True,
        use_pooling_indices: bool = False,
    ):
        if (encoder_dropout_rate > 0.0 or core_dropout_rate > 0.0 or decoder_dropout_rate > 0.0) and (center_dropout_rate > 0.0 or final_dropout_rate > 0.0):
            raise ValueError("Do not specify spatial_dropout together with center_dropout_rate or final_dropout_rate!")
        
        logger.info(
            "Creating UNet with arguments: in_channels=%d, out_channels=%d, num_subnetworks=%d, bilinear=%s, filter_base_count=%d, "
            "center_dropout_rate=%f, final_dropout_rate=%f, "
            "encoder_dropout_rate=%f, core_dropout_rate=%f, decoder_dropout_rate=%f, "
            "use_pooling_indices=%s",
            in_channels,
            out_channels,
            num_subnetworks,
            bilinear,
            filter_base_count,
            center_dropout_rate,
            final_dropout_rate,
            encoder_dropout_rate,
            core_dropout_rate,
            decoder_dropout_rate,
            use_pooling_indices,
        )
        super(MimoUNet, self).__init__()

        self.encoder = SubnetworkEncoder(
            num_subnetworks=num_subnetworks,
            in_channels=in_channels,
            filter_base_count=filter_base_count,
            dropout_rate=encoder_dropout_rate,
            use_pooling_indices=use_pooling_indices,
        )

        self.core = SubnetworkCore(
            num_subnetworks=num_subnetworks,
            filter_base_count=filter_base_count,
            dropout_rate=core_dropout_rate,
            center_dropout_rate=center_dropout_rate,
            bilinear=bilinear,
            use_pooling_indices=use_pooling_indices,
        )

        self.decoder = SubnetworkDecoder(
            num_subnetworks=num_subnetworks,
            filter_base_count=filter_base_count,
            out_channels=out_channels,
            final_dropout_rate=final_dropout_rate,
            dropout_rate=decoder_dropout_rate,
            bilinear=bilinear,
            use_pooling_indices=use_pooling_indices,
        )

    def forward(self, x: torch.Tensor):
        """
        Define the forward pass for MimoUNet.

        Args:
            x: [B, S, C_in, H, W]
        Returns:
            predictions: [B, S, C_out, H, W]
        
        - B: batch size
        - S: number of subnetworks
        - C_in: number of input channels per subnetwork
        - C_out: number of output channels per subnetwork
        - H, W: image dimensions
        """

        x1s, x2s, ind2s = self.encoder(x)
        
        # concatenate along channel dimension
        x2_concat = torch.cat(x2s, axis=1)

        x = self.core(x2_concat)

        return self.decoder(x, x1s, ind2s)
    
class SubnetworkEncoder(nn.Module):
    """
    Subnetwork responsible for encoding input tensor.
    """
    def __init__(
        self, 
        num_subnetworks: int, 
        in_channels: int, 
        filter_base_count: int, 
        dropout_rate: float, 
        use_pooling_indices: bool
    ) -> None:
        super(SubnetworkEncoder, self).__init__()
        
        self.in_convs = create_module_list(
            module=DoubleConv,
            num_subnetworks=num_subnetworks,
            in_channels=in_channels,
            out_channels=filter_base_count,
            dropout_rate=dropout_rate,
        )

        self.down1s = create_module_list(
            module=Down,
            num_subnetworks=num_subnetworks,
            in_channels=filter_base_count,
            out_channels=2 * filter_base_count,
            use_pooling_indices=use_pooling_indices,
            dropout_rate=dropout_rate,
        )

    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Define the forward pass for the encoder.

        Returns:
            Tuple containing lists of tensors for each encoding layer and pooling indices.
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

        return x1s, x2s, ind2s


class SubnetworkCore(nn.Module):
    def __init__(
        self, 
        num_subnetworks: int, 
        filter_base_count: int, 
        dropout_rate: float, 
        center_dropout_rate: float, 
        bilinear: bool,
        use_pooling_indices: bool,
    ) -> None:
        super(SubnetworkCore, self).__init__()

        self.down2 = Down(
            in_channels=2 * filter_base_count * num_subnetworks, 
            out_channels=4 * filter_base_count * num_subnetworks,
            use_pooling_indices=use_pooling_indices,
            dropout_rate=dropout_rate,
        )
        self.down3 = Down(
            in_channels=4 * filter_base_count * num_subnetworks, 
            out_channels=8 * filter_base_count * num_subnetworks,
            use_pooling_indices=use_pooling_indices,
            dropout_rate=dropout_rate,
        )
        self.factor = 2 if (bilinear or use_pooling_indices) else 1
        self.down4 = Down(
            in_channels=8 * filter_base_count * num_subnetworks, 
            out_channels=16 * filter_base_count * num_subnetworks // self.factor, 
            use_pooling_indices=use_pooling_indices,
            dropout_rate=dropout_rate,
        )
        self.center_dropout = nn.Dropout(p=center_dropout_rate)
        self.up1 = Up(
            in_channels=16 * filter_base_count * num_subnetworks, 
            out_channels=8 * filter_base_count * num_subnetworks // self.factor, 
            bilinear=bilinear, 
            use_pooling_indices=use_pooling_indices,
            dropout_rate=dropout_rate,
        )
        self.up2 = Up(
            in_channels=8 * filter_base_count * num_subnetworks, 
            out_channels=4 * filter_base_count * num_subnetworks // self.factor, 
            bilinear=bilinear, 
            use_pooling_indices=use_pooling_indices,
            dropout_rate=dropout_rate,
        )
        self.up3 = Up(
            in_channels=4 * filter_base_count * num_subnetworks, 
            out_channels=2 * filter_base_count * num_subnetworks // self.factor, 
            bilinear=bilinear, 
            use_pooling_indices=use_pooling_indices,
            dropout_rate=dropout_rate,
        )

    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x3, ind3 = self.down2(x)
        x4, ind4 = self.down3(x3)
        x5, ind5 = self.down4(x4)
        x_drop = self.center_dropout(x5)
        x_up = self.up1(x_drop, x4, ind5)
        x_up = self.up2(x_up, x3, ind4)
        x_up = self.up3(x_up, x, ind3)
        return x_up


class SubnetworkDecoder(nn.Module):
    def __init__(
        self,
        num_subnetworks: int,
        filter_base_count: int,
        out_channels: int,
        final_dropout_rate: float,
        dropout_rate: float,
        bilinear: bool,
        use_pooling_indices: bool,
    ) -> None:
        super(SubnetworkDecoder, self).__init__()

        self.num_subnetworks = num_subnetworks
        self.factor = 2 if (bilinear or use_pooling_indices) else 1

        self.up4s = create_module_list(
            module=Up,
            num_subnetworks=num_subnetworks,
            in_channels=2 * filter_base_count * num_subnetworks // self.factor + filter_base_count,
            out_channels=filter_base_count,
            bilinear=bilinear,
            use_pooling_indices=use_pooling_indices,
            dropout_rate=dropout_rate,
        )

        self.final_dropouts = create_module_list(
            module=nn.Dropout,
            num_subnetworks=num_subnetworks,
            p=final_dropout_rate,
        )

        self.outcs = create_module_list(
            module=OutConv,
            num_subnetworks=num_subnetworks,
            in_channels=filter_base_count,
            out_channels=out_channels,
        )

    def forward(
        self, 
        x: torch.Tensor,
        x1s: List[torch.Tensor], 
        ind2s: List[torch.Tensor]
    ) -> torch.Tensor:
        logits = []
        for i in range(self.num_subnetworks):
            x_i = self.up4s[i](x, x1s[i], ind2s[i])
            x_i = self.final_dropouts[i](x_i)
            logits.append(self.outcs[i](x_i))
        
        return torch.stack(logits, axis=1)
