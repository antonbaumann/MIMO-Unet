from argparse import ArgumentParser
from typing import Literal, Dict, Tuple, Any

import pytorch_lightning as pl
import torch

from losses import EvidentialLoss
from metrics import compute_regression_metrics
from utils import count_trainable_parameters
from .mimo_components.model import MimoUNet
from .mimo_components.loss_buffer import LossBuffer


class EvidentialUnetModel(pl.LightningModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            filter_base_count: int,
            center_dropout_rate: float,
            final_dropout_rate: float,
            encoder_dropout_rate: float,
            core_dropout_rate: float,
            decoder_dropout_rate: float,
            weight_decay: float,
            learning_rate: float,
            seed: int,
        ):
        super().__init__()

        # model parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_base_count = filter_base_count
        self.center_dropout_rate = center_dropout_rate
        self.final_dropout_rate = final_dropout_rate
        self.encoder_dropout_rate = encoder_dropout_rate
        self.core_dropout_rate = core_dropout_rate
        self.decoder_dropout_rate = decoder_dropout_rate

        # training parameters
        self.loss_fn = EvidentialLoss(coeff=1.0)
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.seed = seed

        self.model = MimoUNet( 
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_subnetworks=1,
            filter_base_count=self.filter_base_count,
            center_dropout_rate=self.center_dropout_rate,
            final_dropout_rate=self.final_dropout_rate,
            encoder_dropout_rate=self.encoder_dropout_rate,
            core_dropout_rate=self.core_dropout_rate,
            decoder_dropout_rate=self.decoder_dropout_rate,
            bilinear=True,
            use_pooling_indices=False,
        )

        self.save_hyperparameters()
        self.save_hyperparameters({
            "loss": 'evidential',
            "trainable_params": count_trainable_parameters(self.model),
        })

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C_in, H, W]
        Returns:
            out: [B, C_out, H, W]
        """
        B, C_in, H, W = x.shape

        assert C_in == self.in_channels, "channel dimension must match in_channels"

        x = torch.unsqueeze(x, dim=1)
        # [B, C_out, H, W]
        out = self.model(x)
        out = torch.squeeze(out, dim=1)

        mu, logv, logalpha, logbeta = torch.unbind(out, axis=1)

        v = torch.nn.Softplus()(logv) + 1e-5
        alpha = torch.nn.Softplus()(logalpha) + 1 + 1e-5
        beta = torch.nn.Softplus()(logbeta) + 1e-5

        return torch.stack([mu, v, alpha, beta], dim=1)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        image, label = batch["image"], batch["label"]
        mask = batch["mask"] if "mask" in batch else None

        out = self(image)

        loss = self.loss_fn(out, label, mask=mask)
        
        y_pred = self.loss_fn.mode(out).unsqueeze(dim=1)
        aleatoric_std = self.loss_fn.aleatoric_var(out).unsqueeze(dim=1) ** 0.5

        self._log_metrics(y_pred=y_pred, y_true=label, stage="train")

        return {
            "loss": loss.mean(),
            "label": label,
            "preds": y_pred,
            "aleatoric_std_map": aleatoric_std, 
            "err_map": y_pred - label,
            "mask": mask,
        }
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        image, label = batch["image"], batch["label"]
        mask = batch["mask"] if "mask" in batch else None

        out = self(image)

        # [S, ]
        loss = self.loss_fn.forward(out, label, mask=mask, reduce_mean=False)

        # [B, 1, H, W]
        y_pred = self.loss_fn.mode(out).unsqueeze(dim=1)
        aleatoric_std = self.loss_fn.aleatoric_var(out).unsqueeze(dim=1) ** 0.5
        epistemic_std = self.loss_fn.epistemic_var(out).unsqueeze(dim=1) ** 0.5

        self.log("val_loss", loss.mean(), batch_size=self.trainer.datamodule.batch_size)
        self._log_metrics(y_pred=y_pred, y_true=label, stage="val")
        self._log_uncertainties(aleatoric_std, epistemic_std)
       
        return {
            "loss": loss.mean(),
            "label": label,
            "preds": y_pred, 
            "aleatoric_std_map": aleatoric_std, 
            "epistemic_std_map": epistemic_std,
            "err_map": y_pred - label,
            "mask": mask,
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5, verbose=True)
        return dict(
            optimizer=optimizer,
            lr_scheduler=scheduler,
            monitor="val_loss",
        )
    
    def _log_metrics(
            self, 
            y_pred: torch.Tensor, 
            y_true: torch.Tensor,
            stage: Literal["train", "val"] = "train",
        ) -> None:
        """
        Log the metrics for the given stage.
        Args:
            y_pred: [B, C, H, W]
            y_true: [B, C, H, W]
            stage: "train" or "val"
        """
        on_step = stage == "train"
        metric_dict = compute_regression_metrics(
            y_pred.flatten(),
            y_true.flatten(),
        )
        for name, value in metric_dict.items():
            self.log(
                f"metric_{stage}/{name}",
                value,
                on_step=on_step,
                on_epoch=True,
                metric_attribute=name,
                batch_size=self.trainer.datamodule.batch_size,
            )
    
    def _log_uncertainties(self, aleatoric_std: torch.Tensor, epistemic_std: torch.Tensor) -> None:
        self.log("metric_val/aleatoric_std_mean", aleatoric_std.clip(0, 5).mean(), batch_size=self.trainer.datamodule.batch_size)
        self.log("metric_val/epistemic_std_mean", epistemic_std.clip(0, 5).mean(), batch_size=self.trainer.datamodule.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group(title="MIMO UNet Model")
        
        parser.add_argument("--filter_base_count", type=int, default=32)
        parser.add_argument("--center_dropout_rate", type=float, default=0.0)
        parser.add_argument("--final_dropout_rate", type=float, default=0.0)
        parser.add_argument("--encoder_dropout_rate", type=float, default=0.0)
        parser.add_argument("--core_dropout_rate", type=float, default=0.0)
        parser.add_argument("--decoder_dropout_rate", type=float, default=0.0)

        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        return parent_parser
