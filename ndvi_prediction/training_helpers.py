import torch
import logging
import torchvision
import pytorch_lightning as pl

from .visualization import colorize
import numpy as np

logger = logging.getLogger(__name__)


class OutputMonitor(pl.Callback):
    """Log the prediction of the model."""

    def _log_matrix(
        self, 
        img_data: torch.Tensor, 
        log_name: str, 
        global_step: int, 
        pl_module: pl.LightningModule, 
        vmin: float,
        vmax: float,
        cmap: str,
        logger,
    ):
        """Get an batch of images and log them to the tensorboard log.

        img_data: Tensor with the shape batch x vegetation_indices x M x N
        log_name: name of logged image, must contain a formatting placeholder `{veg_index}`
        """
        veg_indices = getattr(pl_module, "target", ["NDVI"])
        veg_indices = [v.lower() for v in veg_indices]
        assert len(veg_indices) == img_data.shape[1], f"Mismatch of veg index count and array shape: {len(veg_indices)} != {img_data.shape[1]}"

        for idx, veg_index in enumerate(veg_indices):
            # img_data shape: Batch x indexes x M x N
            index_data = img_data[:, idx, ...][:, np.newaxis, ...]
            index_grid = torchvision.utils.make_grid(index_data)
            img_color = colorize(index_grid, vmin=vmin, vmax=vmax, cmap=cmap)
            index_log_name = log_name.format(veg_index=veg_index)
            logger.experiment.add_image(
                index_log_name, img_color, dataformats="HWC", global_step=global_step
            )

    def _log_image(
        self,
        img_data: torch.Tensor,
        log_name: str,
        global_step: int,
        pl_module: pl.LightningModule,
        logger,
    ):
        ndvi_kwargs = {"vmin": 0, "vmax": 1, "cmap": "Greens"}
        self._log_matrix(img_data, log_name, global_step, pl_module, logger=logger, **ndvi_kwargs)

    def _log_error_map(
        self, err_map: torch.Tensor, log_name, global_step: int, pl_module: pl.LightningModule, logger
    ):
        vmax = 1
        visualization_kwargs = dict(
            vmin=-2 * vmax,
            vmax=2 * vmax,
            cmap="seismic",
        )
        self._log_matrix(err_map, log_name, global_step, pl_module, logger=logger, **visualization_kwargs)

    def _log_std_map(
        self, std_map: torch.Tensor, log_name, global_step: int, pl_module: pl.LightningModule, logger
    ):
        visualization_kwargs = dict(
            vmin=0.0,
            vmax=1.0,
            cmap="Reds",
        )
        self._log_matrix(std_map, log_name, global_step, pl_module, logger=logger, **visualization_kwargs)


    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            kwargs = {
                "global_step": trainer.global_step,
                "logger": trainer.logger,
                "pl_module": pl_module,
            }
            if isinstance(outputs, list):
                # For GANs, a list with two elements is returned for the Generator and Discriminator
                # but only the Generator contains a tensor as predictions
                _outs = list(filter(lambda l: l["preds"] is not None, outputs))
                assert len(_outs) == 1
                outputs = _outs[0]

            trainer.logger.experiment.add_histogram(
                "train/prediction", outputs["preds"], global_step=trainer.global_step
            )
            self._log_image(
                img_data=outputs["preds"], log_name="train/{veg_index}_predicted", **kwargs
            )
            self._log_image(
                img_data=batch["label"], log_name="train/{veg_index}_true", **kwargs
            )
            self._log_error_map(
                err_map=outputs["err_map"], log_name="train/{veg_index}_error", **kwargs
            )
            self._log_std_map(
                std_map=outputs["std_map"], log_name="train/{veg_index}_std", **kwargs
            )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx % trainer.log_every_n_steps == 0:
            kwargs = {
                "global_step": trainer.global_step,
                "logger": trainer.logger,
                "pl_module": pl_module,
            }

            trainer.logger.experiment.add_histogram(
                "val/prediction", outputs["preds"], global_step=trainer.global_step
            )
            self._log_image(
                img_data=outputs["preds"], log_name="val/{veg_index}_predicted", **kwargs
            )
            self._log_image(
                img_data=batch["label"], log_name="val/{veg_index}_true", **kwargs
            )
            self._log_error_map(
                err_map=outputs["err_map"], log_name="val/{veg_index}_error", **kwargs
            )
            self._log_std_map(
                std_map=outputs["std_map"], log_name="val/{veg_index}_std", **kwargs
            )


class InputMonitor(pl.Callback):
    """Logs the input (input and target label) of the model."""

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx
    ):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            logger = trainer.logger
            logger.experiment.add_histogram(
                "train/input", batch["image"], global_step=trainer.global_step
            )
            logger.experiment.add_histogram(
                "train/target", batch["label"], global_step=trainer.global_step
            )

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if batch_idx % trainer.log_every_n_steps == 0:
            logger = trainer.logger
            logger.experiment.add_histogram(
                "val/input", batch["image"], global_step=trainer.global_step
            )
            logger.experiment.add_histogram(
                "val/target", batch["label"], global_step=trainer.global_step
            )


class LogHparamsMetricCallback(pl.Callback):
    """Log the hp_metric value."""

    def __init__(self, hp_metric_name: str = "val/r2"):
        self.hp_metric_name = hp_metric_name

    def on_validation_end(self, trainer, pl_module):
        if "best_val_r2" in dir (pl_module):
            val_r2_value = trainer.logged_metrics[self.hp_metric_name]
            pl_module.best_val_r2 = torch.max(
                val_r2_value, pl_module.best_val_r2.to(val_r2_value.device)
            )
            trainer.logger.experiment.add_scalar(
                "hp_metric", pl_module.best_val_r2, global_step=trainer.global_step
            )
