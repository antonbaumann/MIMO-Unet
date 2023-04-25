from enum import Enum

from argparse import ArgumentParser
import logging
from pathlib import Path
from typing import Optional, List

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torchmetrics

from ndvi_prediction.training_helpers import InputMonitor, OutputMonitor, LogHparamsMetricCallback
import models

import sen12tp
import sen12tp.constants
import sen12tp.dataset
import sen12tp.utils
import sen12tp.zarr_dataset
import sen12tp.tif_dataset
from sen12tp.dataset import Patchsize

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_metrics_dict(mode: str = "train", vegetation_index: str = "", metrics: Optional[List[str]] = None):
    if metrics is None:
        metrics = ["MAE", "R2", "MSE"]
    seperator = "/" if vegetation_index else ""
    metrics_dict = dict()
    if "MAE" in metrics:
        metrics_dict[f"{mode}/{vegetation_index}{seperator}mae"] = torchmetrics.MeanAbsoluteError()
    if "R2" in metrics:
        metrics_dict[f"{mode}/{vegetation_index}{seperator}r2"] = torchmetrics.R2Score()
    if "MSE" in metrics:
        metrics_dict[f"{mode}/{vegetation_index}{seperator}mse"] = torchmetrics.MeanSquaredError()
    if "SSIM" in metrics:
        metrics_dict[f"{mode}/{vegetation_index}{seperator}ssim"] = torchmetrics.StructuralSimilarityIndexMeasure()
    if "MultiscaleSSIM" in metrics:
        metrics_dict[f"{mode}/{vegetation_index}{seperator}multiscalessim"] = torchmetrics.MultiScaleStructuralSimilarityIndexMeasure()
    if mode == "val":
        pass
       # metrics_dict[f"{mode}/{vegetation_index}{seperator}pearson_corr"] = torchmetrics.PearsonCorrcoef()
    return metrics_dict


def get_datamodule(args_dict: dict) -> sen12tp.dataset.BaseDataModule:
    args_dict = args_dict.copy()

    if args_dict["data_module"] == "tif":
        #args_dict["patch_size"] = 256
        patchsize = Patchsize(args_dict["patch_size"], args_dict["patch_size"])
        if "patch_size" in args_dict:
            del args_dict["patch_size"]  # to avoid a duplicate kwarg 'patch_size'
        dm = sen12tp.tif_dataset.Sen12TpDataModule(
            patch_size=patchsize,
            model_inputs=args_dict["input"],
            model_targets=args_dict["target"],
            transform=sen12tp.utils.min_max_transform,
            **args_dict,
        )
    elif args_dict["data_module"] == "zarr":
        dm = sen12tp.zarr_dataset.ZarrDataModule(
            model_input=args_dict["input"],
            model_target=args_dict["target"],
            **args_dict
        )
    else:
        raise ValueError("specified data_module not available!", args_dict["data_module"])
    dm.setup()
    return dm


def dir_path(string) -> Path:
    """Helper for argument parsing. Ensures that the provided string is a directory."""
    path = Path(string)
    if path.is_dir():
        return path
    else:
        raise NotADirectoryError(string)


def get_argument_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=dir_path,
        default="/scratch/datasets/sen12tp-cropland-splitted/",
    )
    parser.add_argument("--checkpoint_path", type=dir_path, required=True,
                        help="Path where the lightning logs and checkpoints should be saved to.")
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Specify the maximum number of epochs to train.",
    )
    parser.add_argument("--limit_train_batches", type=int, required=False, default=1.0)
    parser.add_argument("--limit_val_batches", type=int, required=False, default=1.0)
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use for training.")
    parser = sen12tp.dataset.BaseDataModule.add_model_specific_args(parser)

    # get model specific command line arguments
    for model_name in dir(models):
        model_cls = getattr(models, model_name)
        if "add_model_specific_args" in dir(model_cls):
           parser = model_cls.add_model_specific_args(parser)
    return parser


def get_default_callbacks(validation: bool = True) -> List[pl.Callback]:
    callbacks = [
        # VisualInspectionROIs(),
        InputMonitor(),
        OutputMonitor(),
        LogHparamsMetricCallback(),
        ModelCheckpoint(save_last=True),
        # TelegramProgressBar(),
    ]
    if validation:
        callbacks_validation = [
            ModelCheckpoint(
                monitor="metric_val/mae_epoch",
                save_top_k=5,
                filename="epoch-{epoch}-step-{step}-valloss-{val_loss:.8f}-mae-{metric_val/mae_epoch:.8f}",
                auto_insert_metric_name=False,
            ),
            EarlyStopping(monitor="metric_val/mae_epoch", patience=10, mode="min", min_delta=0.00025),
        ]
        callbacks += callbacks_validation
    return callbacks
