from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

from losses import UncertaintyLoss
from metrics import compute_regression_metrics
from .mimo_components.model import MimoUNet


class MimoUnetModel(pl.LightningModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_subnetworks: int,
            filter_base_count: int,
            center_dropout_rate: float,
            final_dropout_rate: float,
            loss: str,
            weight_decay: float,
            learning_rate: float,
            seed: int,
            input_repetition_probability: float = 0.0,
            batch_repetitions: int = 1,
        ):
        super().__init__()

        # model parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subnetworks = num_subnetworks
        self.filter_base_count = filter_base_count
        self.center_dropout_rate = center_dropout_rate
        self.final_dropout_rate = final_dropout_rate

        # training parameters
        self.loss_fn = UncertaintyLoss.from_name(loss)
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.seed = seed
        self.input_repetition_probability = input_repetition_probability
        self.batch_repetitions = batch_repetitions

        self.model = MimoUNet( 
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_subnetworks=self.num_subnetworks,
            filter_base_count=self.filter_base_count,
            center_dropout_rate=self.center_dropout_rate,
            final_dropout_rate=self.final_dropout_rate,
            bilinear=True,
            use_pooling_indices=False,
        )

        self.save_hyperparameters()
        self.save_hyperparameters({"loss_name": loss})


    def _apply_input_transform(
            self, 
            x: torch.Tensor,
        ):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            x: [B, S, C, H, W]
        """
        B, _, _, _ = x.shape

        main_shuffle = torch.randperm(B).repeat(self.batch_repetitions)
        to_shuffle = int(main_shuffle.shape[0] * (1. - self.input_repetition_probability))

        shuffle_indices = [
            torch.cat(
                (main_shuffle[:to_shuffle].random_(to_shuffle), main_shuffle[to_shuffle:]), 
                dim=0,
            ) for _ in range(self.num_subnetworks)
        ]

        return torch.stack(
            [torch.index_select(x, 0, indices) for indices in shuffle_indices], 
            dim=1,
        )

    def _reshape_for_subnetwors(self, x: torch.Tensor, repeat: bool = False):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            x: [B // S, S, C, H, W]
        """
        B, C, H, W = x.shape

        # [B, C, H, W] -> [B // S, S, C, H, W]
        if not repeat:
            assert B % self.num_subnetworks == 0, "batch dimension must be divisible by num_subnetworks"
            return x.view(B // self.num_subnetworks, self.num_subnetworks, C, H, W)
        else:
            x = x[:, None, :, :, :]
            return x.repeat(1, self.num_subnetworks, 1, 1, 1)

    def _reshape_for_plotting(self, x: torch.Tensor):
        """
        Args:
            x: [B // S, S, C, H, W]
        Returns:
            x: [B, C, H, W]
        """
        B_, S, C, H, W = x.shape
        # [B // S, S, C, H, W] -> [B, C, H, W]
        return x.view(B_ * S, C, H, W)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, S, C_in, H, W]
        Returns:
            p1: [B, S, C_out, H, W]
            p2: [B, S, C_out, H, W]
        """
        B, S, C_in, H, W = x.shape

        assert S == self.num_subnetworks, "subnetwork dimension must match num_subnetworks"
        assert C_in == self.in_channels, "channel dimension must match in_channels"

        # [B, S, 2*C_out, H, W]
        out = self.model(x)

        # [B, S, C_out, H, W]
        p1 = out[:, :, :self.out_channels // 2, ...]
        p2 = out[:, :, self.out_channels // 2:, ...]

        return p1, p2

    def training_step(self, batch, batch_idx):
        image, label = batch["image"], batch["label"]
        
        image_transformed = self._apply_input_transform(image)
        label_transformed = self._apply_input_transform(label)

        p1, p2 = self(image_transformed)

        loss = self.loss_fn.forward(p1, p2, label_transformed, reduce_mean=True)
        y_hat = self.loss_fn.mode(p1, p2)
        aleatoric_std = self.loss_fn.std(p1, p2)

        self.log("train_loss", loss, batch_size=self.trainer.datamodule.batch_size)
        metric_dict = compute_regression_metrics(
            y_hat.flatten(), 
            label_transformed.flatten(),
        )

        for name, value in metric_dict.items():
            self.log(
                f"metric_train/{name}",
                value,
                on_step=True,
                on_epoch=True,
                metric_attribute=name,
                batch_size=self.trainer.datamodule.batch_size,
            )

        return {
            "loss": loss,
            "preds": self._reshape_for_plotting(y_hat), 
            "aleatoric_std_map": self._reshape_for_plotting(aleatoric_std), 
            "err_map": self._reshape_for_plotting(y_hat - label_transformed),
        }
    
    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]

        x = self._reshape_for_subnetwors(x, repeat=True)
        y = self._reshape_for_subnetwors(y, repeat=True)

        p1, p2 = self(x)

        # [S, ]
        val_loss = self.loss_fn.forward(p1, p2, y, reduce_mean=False).mean(dim=(0, 2, 3, 4))

        # [B, S, 1, H, W]
        y_hat = self.loss_fn.mode(p1, p2)
        aleatoric_std = self.loss_fn.std(p1, p2).mean(dim=1)
        y_hat_mean = y_hat.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1)

        if self.num_subnetworks == 1:
            epistemic_std = torch.zeros_like(aleatoric_std)
        else:
            epistemic_std = (torch.sum((y_hat - y_hat_mean) ** 2, dim=1) * (1 / (self.num_subnetworks - 1))) ** 0.5

        self.log("val_loss", val_loss.mean(), batch_size=self.trainer.datamodule.batch_size)
        for subnetwork_idx in range(val_loss.shape[0]):
            self.log(f"val_loss_{subnetwork_idx}", val_loss[subnetwork_idx], batch_size=self.trainer.datamodule.batch_size)

        metric_dict = compute_regression_metrics(
            y_hat_mean.flatten(), 
            y_mean.flatten(),
        )
        for name, value in metric_dict.items():
            self.log(
                f"metric_val/{name}",
                value,
                on_step=True,
                on_epoch=True,
                metric_attribute=name,
                batch_size=self.trainer.datamodule.batch_size,
            )

        return {
            "loss": val_loss.mean(), 
            "preds": y_hat_mean.squeeze(dim=1), 
            "aleatoric_std_map": aleatoric_std, 
            "epistemic_std_map": epistemic_std,
            "err_map": y_hat_mean.squeeze(dim=1) - y_mean,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.5, verbose=True)
        return [optimizer], [scheduler]
    
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group(title="MIMO UNet Model")
        
        parser.add_argument("--num_subnetworks", type=int, default=3)
        parser.add_argument("--filter_base_count", type=int, default=32)
        parser.add_argument("--center_dropout_rate", type=float, default=0)
        parser.add_argument("--final_dropout_rate", type=float, default=0)

        parser.add_argument("--input_repetition_probability", type=float, default=0.0)
        parser.add_argument("--batch_repetitions", type=int, default=1)
        parser.add_argument("--loss", type=str, default="laplace_nll")
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=0)
        return parent_parser
