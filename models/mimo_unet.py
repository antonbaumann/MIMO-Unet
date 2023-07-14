from argparse import ArgumentParser
from typing import Literal, Dict, Tuple, Any

import pytorch_lightning as pl
import torch

from losses import UncertaintyLoss
from metrics import compute_regression_metrics
from utils import count_trainable_parameters
from .mimo_components.model import MimoUNet
from .mimo_components.loss_buffer import LossBuffer
from .utils import repeat_subnetworks, apply_input_transform, compute_uncertainties


class MimoUnetModel(pl.LightningModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_subnetworks: int,
            filter_base_count: int,
            center_dropout_rate: float,
            final_dropout_rate: float,
            encoder_dropout_rate: float,
            core_dropout_rate: float,
            decoder_dropout_rate: float,
            loss: str,
            weight_decay: float,
            learning_rate: float,
            seed: int,
            loss_buffer_size: int,
            loss_buffer_temperature: float,
            input_repetition_probability: float = 0.0,
            batch_repetitions: int = 1,
            scheduler_step_size: int = 20,
            scheduler_gamma: float = 0.5,
        ):
        super().__init__()

        # model parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subnetworks = num_subnetworks
        self.filter_base_count = filter_base_count
        self.center_dropout_rate = center_dropout_rate
        self.final_dropout_rate = final_dropout_rate
        self.encoder_dropout_rate = encoder_dropout_rate
        self.core_dropout_rate = core_dropout_rate
        self.decoder_dropout_rate = decoder_dropout_rate

        # training parameters
        self.loss_fn = UncertaintyLoss.from_name(loss)
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.seed = seed
        self.loss_buffer_size = loss_buffer_size
        self.loss_buffer_temperature = loss_buffer_temperature
        self.input_repetition_probability = input_repetition_probability
        self.batch_repetitions = batch_repetitions
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma

        self.model = MimoUNet( 
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_subnetworks=self.num_subnetworks,
            filter_base_count=self.filter_base_count,
            center_dropout_rate=self.center_dropout_rate,
            final_dropout_rate=self.final_dropout_rate,
            encoder_dropout_rate=self.encoder_dropout_rate,
            core_dropout_rate=self.core_dropout_rate,
            decoder_dropout_rate=self.decoder_dropout_rate,
            bilinear=True,
            use_pooling_indices=False,
        )

        self.loss_buffer = LossBuffer(
            buffer_size=self.loss_buffer_size,
            temperature=self.loss_buffer_temperature,
            subnetworks=self.num_subnetworks,
        )

        self.save_hyperparameters()
        self.save_hyperparameters({
            "loss": loss,
            "trainable_params": count_trainable_parameters(self.model),
        })

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
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        image, label = batch["image"], batch["label"]
        mask = batch["mask"] if "mask" in batch else None
        
        image_transformed, label_transformed, mask_transformed = apply_input_transform(
            image, 
            label, 
            mask,
            num_subnetworks=self.num_subnetworks,
            input_repetition_probability=self.input_repetition_probability,
            batch_repetitions=self.batch_repetitions,
        )

        p1, p2 = self(image_transformed)
        y_pred = self.loss_fn.mode(p1, p2)
        aleatoric_std = self.loss_fn.std(p1, p2)

        loss, loss_weighted, weights = self._calculate_train_loss(p1, p2, y_true=label_transformed, mask=mask_transformed)

        self._log_train_loss_and_weights(loss, weights)
        self._log_metrics(y_pred=y_pred, y_true=label_transformed, stage="train")

        return {
            "loss": loss_weighted.mean(),
            "label": self._flatten_subnetwork_dimension(label_transformed),
            "preds": self._flatten_subnetwork_dimension(y_pred),
            "aleatoric_std_map": self._flatten_subnetwork_dimension(aleatoric_std), 
            "err_map": self._flatten_subnetwork_dimension(y_pred - label_transformed),
            "mask": self._flatten_subnetwork_dimension(mask_transformed) if mask_transformed is not None else None,
        }
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        image, label = batch["image"], batch["label"]
        mask = batch["mask"] if "mask" in batch else None

        image = repeat_subnetworks(image, num_subnetworks=self.num_subnetworks)
        label = repeat_subnetworks(label, num_subnetworks=self.num_subnetworks)
        mask_transformed = repeat_subnetworks(mask, num_subnetworks=self.num_subnetworks) if mask is not None else None

        p1, p2 = self(image)

        # [S, ]
        val_loss = self.loss_fn.forward(p1, p2, label, mask=mask_transformed, reduce_mean=False).mean(dim=(0, 2, 3, 4))

        # [B, S, H, W]
        y_pred_mean, aleatoric_var, epistemic_var = compute_uncertainties(self.loss_fn, p1, p2)
        y_mean = label.mean(dim=1)

        combined_var = aleatoric_var + epistemic_var
        combined_std = torch.sqrt(combined_var)
        aleatoric_std = torch.sqrt(aleatoric_var)
        epistemic_std = torch.sqrt(epistemic_var)

        combined_log_scale = self.loss_fn.calculate_dist_param(std=combined_std, log=True)
        val_loss_combined = self.loss_fn.forward(p1.mean(dim=1), combined_log_scale, y_mean, mask=mask, reduce_mean=True)

        self._log_val_loss(val_loss, val_loss_combined)
        self._log_metrics(y_pred=y_pred_mean, y_true=y_mean, stage="val")
        self._log_uncertainties(aleatoric_std, epistemic_std)
       
        return {
            "loss": val_loss.mean(),
            "label": y_mean,
            "preds": y_pred_mean, 
            "aleatoric_std_map": aleatoric_std, 
            "epistemic_std_map": epistemic_std,
            "err_map": y_pred_mean - y_mean,
            "mask": mask,
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma, 
            verbose=True,
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=scheduler,
            monitor="val_loss",
        )

    @staticmethod
    def _compute_epistemic_std(y_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute the epistemic uncertainty from a set of predictions.

        Args:
            y_hat: [B, S, C, H, W]
        Returns:
            [B, C, H, W]
        """
        B, S, C, H, W = y_hat.shape

        if S == 1:
            return torch.zeros((B, C, H, W), device=y_hat.device)
        
        y_hat_mean = y_hat.mean(dim=1, keepdim=True)
        normalizing_const = 1 / (S - 1)
        variance = torch.sum((y_hat - y_hat_mean) ** 2, dim=1) * normalizing_const
        return variance ** 0.5

    def _calculate_train_loss(
            self, 
            p1: torch.Tensor, 
            p2: torch.Tensor, 
            y_true: torch.Tensor,
            mask: torch.Tensor = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the (weighted) training loss.
        
        Args:
            p1, p2, y_true: Input tensors of shape [B, S, C, H, W]
            mask: Optional mask tensor
        
        Returns:
            loss, loss_weighted, weights: Output tensors of shape [S, ]
        """

        forward = self.loss_fn.forward(p1, p2, y_true, reduce_mean=False, mask=mask)
        loss = forward.mean(dim=(0, 2, 3, 4))
        weights = self.loss_buffer.get_weights().to(loss.device)

        self.loss_buffer.add(loss.detach())

        return loss, loss * weights, weights
    
    def _log_train_loss_and_weights(self, loss: torch.Tensor, weights: torch.Tensor) -> None:
        self.log("train_loss", loss.mean(), batch_size=self.trainer.datamodule.batch_size)
        for subnetwork_idx in range(loss.shape[0]):
            self.log(f"train_loss_{subnetwork_idx}", loss[subnetwork_idx], batch_size=self.trainer.datamodule.batch_size)
            self.log(f"train_weight_{subnetwork_idx}", weights[subnetwork_idx], batch_size=self.trainer.datamodule.batch_size)
    
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
    
    def _log_val_loss(self, val_loss: torch.Tensor, val_loss_combined: torch.Tensor) -> None:
        self.log("val_loss", val_loss.mean(), batch_size=self.trainer.datamodule.batch_size)
        for subnetwork_idx in range(val_loss.shape[0]):
            self.log(f"val_loss_{subnetwork_idx}", val_loss[subnetwork_idx], batch_size=self.trainer.datamodule.batch_size)
        self.log("val_loss_combined", val_loss_combined, batch_size=self.trainer.datamodule.batch_size)
    
    def _log_uncertainties(self, aleatoric_std: torch.Tensor, epistemic_std: torch.Tensor) -> None:
        self.log("metric_val/aleatoric_std_mean", aleatoric_std.clip(0, 5).mean(), batch_size=self.trainer.datamodule.batch_size)
        self.log("metric_val/epistemic_std_mean", epistemic_std.clip(0, 5).mean(), batch_size=self.trainer.datamodule.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group(title="MIMO UNet Model")
        
        parser.add_argument("--num_subnetworks", type=int, default=3)
        parser.add_argument("--filter_base_count", type=int, default=32)
        parser.add_argument("--center_dropout_rate", type=float, default=0.0)
        parser.add_argument("--final_dropout_rate", type=float, default=0.0)
        parser.add_argument("--encoder_dropout_rate", type=float, default=0.0)
        parser.add_argument("--core_dropout_rate", type=float, default=0.0)
        parser.add_argument("--decoder_dropout_rate", type=float, default=0.0)

        parser.add_argument("--input_repetition_probability", type=float, default=0.0)
        parser.add_argument("--batch_repetitions", type=int, default=1)
        parser.add_argument("--loss", type=str, default="laplace_nll")
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--loss_buffer_size", type=int, default=10)
        parser.add_argument("--loss_buffer_temperature", type=float, default=1.0)
        parser.add_argument("--scheduler_step_size", type=int, default=20)
        parser.add_argument("--scheduler_gamma", type=float, default=0.5)
        return parent_parser
