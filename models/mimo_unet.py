from argparse import ArgumentParser
from typing import Literal, Dict, Tuple, Any

import pytorch_lightning as pl
import torch

from losses import UncertaintyLoss
from metrics import compute_regression_metrics
from utils import count_trainable_parameters
from .mimo_components.model import MimoUNet
from .mimo_components.loss_buffer import LossBuffer


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
        self.input_repetition_probability = input_repetition_probability
        self.batch_repetitions = batch_repetitions
        self.loss_buffer_size = loss_buffer_size
        self.loss_buffer_temperature = loss_buffer_temperature

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

        print(image.shape, label.shape, mask.shape)
        
        image_transformed, label_transformed, mask_transformed = self._apply_input_transform(image, label, mask)

        p1, p2 = self(image_transformed)
        y_pred = self.loss_fn.mode(p1, p2)
        aleatoric_std = self.loss_fn.std(p1, p2)

        print(p1.shape, p2.shape, y_pred.shape, aleatoric_std.shape, mask_transformed.shape)
        loss, loss_weighted, weights = self._calculate_train_loss(p1, p2, y_true=label_transformed, mask=mask_transformed)

        print(loss.shape, loss_weighted.shape, weights.shape)

        self._log_train_loss_and_weights(loss, weights)
        self._log_metrics(y_pred=y_pred, y_true=label_transformed, stage="train")

        return {
            "loss": loss_weighted.mean(),
            "label": self._flatten_subnetwork_dimension(label_transformed),
            "preds": self._flatten_subnetwork_dimension(y_pred), 
            "aleatoric_std_map": self._flatten_subnetwork_dimension(aleatoric_std), 
            "err_map": self._flatten_subnetwork_dimension(y_pred - label_transformed),
            "mask": self._flatten_subnetwork_dimension(mask_transformed),
        }
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch["image"], batch["label"]
        mask = batch["mask"] if "mask" in batch else None

        x = self._repeat_subnetworks(x)
        y = self._repeat_subnetworks(y)
        mask_transformed = self._repeat_subnetworks(mask) if mask is not None else None

        p1, p2 = self(x)

        # [S, ]
        val_loss = self.loss_fn.forward(p1, p2, y, mask=mask_transformed, reduce_mean=False).mean(dim=(0, 2, 3, 4))

        # [B, S, 1, H, W]
        y_pred = self.loss_fn.mode(p1, p2)
        aleatoric_std = self.loss_fn.std(p1, p2).mean(dim=1)
        y_pred_mean = y_pred.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1)

        epistemic_std = self._compute_epistemic_std(y_pred)
        combined_std = (aleatoric_std ** 2 + epistemic_std ** 2) ** 0.5
        combined_log_scale = self.loss_fn.calculate_dist_param(std=combined_std, log=True)

        val_loss_combined = self.loss_fn.forward(p1.mean(dim=1), combined_log_scale, y_mean, mask=mask, reduce_mean=True)

        self._log_val_loss(val_loss, val_loss_combined)
        self._log_metrics(y_pred=y_pred_mean, y_true=y_mean, stage="val")
       
        return {
            "loss": val_loss.mean(),
            "label": y_mean,
            "preds": y_pred_mean.squeeze(dim=1), 
            "aleatoric_std_map": aleatoric_std, 
            "epistemic_std_map": epistemic_std,
            "err_map": y_pred_mean.squeeze(dim=1) - y_mean,
            "mask": mask,
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.5, verbose=True)
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

    def _apply_input_transform(
            self, 
            image: torch.Tensor,
            label: torch.Tensor,
            mask: torch.Tensor = None,
        ):
        """
        Apply input transformations to the input data.
        - batch repetition
        - input repetition

        Args:
            image: [B, C_image, H, W]
            label: [B, C_label, H, W]
        Returns:
            [B, S, C_image, H, W], [B, S, C_label, H, W]
        """
        B, _, _, _ = image.shape

        main_shuffle = torch.randperm(B, device=image.device).repeat(self.batch_repetitions)
        to_shuffle = int(main_shuffle.shape[0] * (1. - self.input_repetition_probability))

        shuffle_indices = [
            torch.cat(
                (main_shuffle[:to_shuffle][torch.randperm(to_shuffle)], main_shuffle[to_shuffle:]), 
                dim=0,
            ) for _ in range(self.num_subnetworks)
        ]

        image_transformed = torch.stack(
            [torch.index_select(image, 0, indices) for indices in shuffle_indices], 
            dim=1,
        )
        label_transformed = torch.stack(
            [torch.index_select(label, 0, indices) for indices in shuffle_indices],
            dim=1,
        )
        mask_transformed = torch.stack(
            [torch.index_select(mask, 0, indices) for indices in shuffle_indices],
            dim=1,
        ) if mask is not None else None
        return image_transformed, label_transformed, mask_transformed

    def _repeat_subnetworks(self, x: torch.Tensor):
        """
        Repeat the input tensor along the subnetwork dimension.

        Args:
            x: [B, C, H, W]
        Returns:
            x: [B, S, C, H, W]
        """
        x = x[:, None, :, :, :]
        return x.repeat(1, self.num_subnetworks, 1, 1, 1)

    def _flatten_subnetwork_dimension(self, x: torch.Tensor):
        """
        Collapse the subnetwork dimension into the batch dimension.

        Args:
            x: [B // S, S, C, H, W]
        Returns:
            x: [B, C, H, W]
        """
        B_, S, C, H, W = x.shape
        # [B // S, S, C, H, W] -> [B, C, H, W]
        return x.view(B_ * S, C, H, W)

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
            p1: [B, S, C, H, W]
            p2: [B, S, C, H, W]
            y_true: [B, S, C, H, W]
        Returns:
            loss, loss_weighted, weights: [S, ], [S, ], [S, ]
        """

        forward = self.loss_fn.forward(p1, p2, y_true, reduce_mean=False, mask=mask)
        print(forward.shape)
        loss = forward.mean(dim=(0, 2, 3, 4))
        print(loss.shape)
        weights = self.loss_buffer.get_weights().to(loss.device)
        print(weights.shape)
        loss_weighted = loss * weights
        self.loss_buffer.add(loss.detach())
        return loss, loss_weighted, weights
    
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
        return parent_parser
