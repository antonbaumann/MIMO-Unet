from typing import Optional
import torch
import torchvision
import pytorch_lightning as pl
import wandb

from visualization import colorize
import numpy as np


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

        images = wandb.Image(img_color)
        wandb.log({log_name: images}, step=global_step)

    def _log_image(
        self,
        img_data: torch.Tensor,
        log_name: str,
        global_step: int,
        pl_module: pl.LightningModule,
        logger,
    ):
        ndvi_kwargs = {"vmin": 0, "cmap": "Greys"}
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

            self._log_image(
                img_data=outputs["preds"], log_name="train/depth_predicted", **kwargs
            )
            self._log_image(
                img_data=outputs["label"], log_name="train/depth_true", **kwargs
            )
            self._log_error_map(
                err_map=outputs["err_map"], log_name="train/depth_error", **kwargs
            )
            self._log_std_map(
                std_map=outputs["aleatoric_std_map"], log_name="train/depth_aleatoric_std", **kwargs
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

            self._log_image(
                img_data=outputs["preds"], log_name="val/depth_predicted", **kwargs
            )
            self._log_image(
                img_data=outputs["label"], log_name="val/depth_true", **kwargs
            )
            self._log_error_map(
                err_map=outputs["err_map"], log_name="val/depth_error", **kwargs
            )
            self._log_std_map(
                std_map=outputs["aleatoric_std_map"], log_name="val/depth_aleatoric_std", **kwargs
            )
            if 'epistemic_std_map' in outputs:
                self._log_std_map(
                    std_map=outputs["epistemic_std_map"], log_name="val/depth_epistemic_std", **kwargs
                )
