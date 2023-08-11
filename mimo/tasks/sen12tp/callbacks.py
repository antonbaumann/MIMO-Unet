from typing import Optional, List
import warnings
import torch
import torchvision
import lightning.pytorch as pl
import wandb
import numpy as np

from mimo.visualization import colorize


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
        max_images: int = 32,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ):
        """Get an batch of images and log them to the tensorboard log.

        img_data: Tensor with the shape batch x vegetation_indices x M x N
        log_name: name of logged image, must contain a formatting placeholder `{veg_index}`
        """
        veg_indices = pl_module.trainer.datamodule.model_targets
        assert len(veg_indices) == img_data.shape[1], f"Mismatch of veg index count and array shape: {len(veg_indices)} != {img_data.shape[1]}"

        for idx, veg_index in enumerate(veg_indices):
            # img_data shape: Batch x indexes x M x N
            index_data = img_data[:max_images, idx, ...][:, np.newaxis, ...]
            index_grid = torchvision.utils.make_grid(index_data)
            img_color = colorize(index_grid, vmin=vmin, vmax=vmax, cmap=cmap)
            index_log_name = log_name.format(veg_index=veg_index)

            if isinstance(logger, pl.loggers.wandb.WandbLogger):
                images = wandb.Image(img_color)
                wandb.log({index_log_name: images}, step=global_step)
            if isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
                logger.experiment.add_image(index_log_name, img_color, dataformats="HWC", global_step=global_step)
            else:
                warnings.warn(f"Logger {logger} not supported for logging images.")
            

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

            # trainer.logger.experiment.add_histogram(
            #     "train/prediction", outputs["preds"], global_step=trainer.global_step
            # )
            self._log_image(
                img_data=outputs["preds"], log_name="train/{veg_index}_predicted", **kwargs
            )
            self._log_image(
                img_data=outputs["label"], log_name="train/{veg_index}_true", **kwargs
            )
            self._log_error_map(
                err_map=outputs["err_map"], log_name="train/{veg_index}_error", **kwargs
            )
            self._log_std_map(
                std_map=outputs["aleatoric_std_map"], log_name="train/{veg_index}_aleatoric_std", **kwargs
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
                img_data=outputs["preds"], log_name="val/{veg_index}_predicted", **kwargs
            )
            self._log_image(
                img_data=outputs["label"], log_name="val/{veg_index}_true", **kwargs
            )
            self._log_error_map(
                err_map=outputs["err_map"], log_name="val/{veg_index}_error", **kwargs
            )
            self._log_std_map(
                std_map=outputs["aleatoric_std_map"], log_name="val/{veg_index}_aleatoric_std", **kwargs
            )
            if 'epistemic_std_map' in outputs:
                self._log_std_map(
                    std_map=outputs["epistemic_std_map"], log_name="val/{veg_index}_epistemic_std", **kwargs
                )
