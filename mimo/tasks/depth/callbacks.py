from typing import Optional
import warnings
import torch
import torchvision
import lightning.pytorch as pl
import wandb
import numpy as np

from mimo.visualization import colorize


class WandbMetricsDefiner(pl.Callback):
    def on_fit_start(self, trainer, pl_module):
        wandb.define_metric('metric_val/r2', summary='max')
        wandb.define_metric('metric_val/mae', summary='min')
        wandb.define_metric('metric_val/mse', summary='min')

class OutputMonitor(pl.Callback):
    """Log the prediction of the model."""

    def _log_matrix(
        self, 
        img_data: torch.Tensor, 
        log_name: str, 
        global_step: int, 
        pl_module: pl.LightningModule, 
        cmap: str,
        logger,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        max_images: int = 32,
    ):
        """Get an batch of images and log them to the tensorboard log.

        img_data: Tensor with the shape batch x vegetation_indices x M x N
        log_name: name of logged image, must contain a formatting placeholder `{veg_index}`
        """
        index_data = img_data[:max_images, 0, ...][:, np.newaxis, ...]
        index_grid = torchvision.utils.make_grid(index_data)
        img_color = colorize(index_grid, vmin=vmin, vmax=vmax, cmap=cmap)

        if isinstance(logger, pl.loggers.wandb.WandbLogger):
            images = wandb.Image(img_color)
            wandb.log({log_name: images}, step=global_step)
        if isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
            logger.experiment.add_image(log_name, img_color, dataformats="HWC", global_step=global_step)
        else:
            warnings.warn(f"Logger {logger} not supported for logging images.")

    def _log_image(
        self,
        img_data: torch.Tensor,
        log_name: str,
        global_step: int,
        pl_module: pl.LightningModule,
        logger,
        mask: Optional[torch.Tensor] = None,
    ):
        ndvi_kwargs = {"vmin": 0, "vmax": 1, "cmap": "turbo"}
        if mask is not None:
            img_data = img_data * mask
        self._log_matrix(img_data, log_name, global_step, pl_module, logger=logger, **ndvi_kwargs)

    def _log_error_map(
        self, err_map: torch.Tensor, log_name, global_step: int, pl_module: pl.LightningModule, logger, mask: Optional[torch.Tensor] = None
    ):
        err_map = torch.abs(err_map)
        visualization_kwargs = dict(
            vmin=0,
            vmax=2,
            cmap="Reds",
        )
        if mask is not None:
            err_map = err_map * mask
        self._log_matrix(err_map, log_name, global_step, pl_module, logger=logger, **visualization_kwargs)

    def _log_std_map(
        self, std_map: torch.Tensor, log_name, global_step: int, pl_module: pl.LightningModule, logger, mask: Optional[torch.Tensor] = None
    ):
        visualization_kwargs = dict(
            vmin=0.0,
            vmax=1.0,
            cmap="Reds",
        )
        if mask is not None:
            std_map = std_map * mask
        self._log_matrix(std_map, log_name, global_step, pl_module, logger=logger, **visualization_kwargs)


    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx,
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

            self._log_image(
                img_data=outputs["preds"], mask=outputs.get('mask', None), log_name="train/depth_predicted", **kwargs
            )
            self._log_image(
                img_data=outputs["label"], mask=outputs.get('mask', None), log_name="train/depth_true", **kwargs
            )
            self._log_error_map(
                err_map=outputs["err_map"], mask=outputs.get('mask', None), log_name="train/depth_error", **kwargs
            )
            self._log_std_map(
                std_map=outputs["aleatoric_std_map"], mask=outputs.get('mask', None), log_name="train/depth_aleatoric_std", **kwargs
            )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx % trainer.log_every_n_steps == 0:
            kwargs = {
                "global_step": trainer.global_step,
                "logger": trainer.logger,
                "pl_module": pl_module,
            }

            self._log_image(
                img_data=outputs["preds"], mask=outputs.get('mask', None), log_name="val/depth_predicted", **kwargs
            )
            self._log_image(
                img_data=outputs["label"], mask=outputs.get('mask', None), log_name="val/depth_true", **kwargs
            )
            self._log_error_map(
                err_map=outputs["err_map"], mask=outputs.get('mask', None), log_name="val/depth_error", **kwargs
            )
            self._log_std_map(
                std_map=outputs["aleatoric_std_map"], mask=outputs.get('mask', None), log_name="val/depth_aleatoric_std", **kwargs
            )
            if 'epistemic_std_map' in outputs:
                self._log_std_map(
                    std_map=outputs["epistemic_std_map"], mask=outputs.get('mask', None), log_name="val/depth_epistemic_std", **kwargs
                )
