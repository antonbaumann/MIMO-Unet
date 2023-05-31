from typing import List
from argparse import Namespace, ArgumentParser
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from utils import dir_path
from models.mimo_unet import MimoUnetModel
from tasks.depth.nyuv2_datamodule import NYUv2DepthDataModule
from tasks.depth.callbacks import OutputMonitor, WandbMetricsDefiner

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def default_callbacks(validation: bool = True) -> List[pl.Callback]:
    callbacks = [
        OutputMonitor(),
        ModelCheckpoint(save_last=True),
    ]
    if validation:
        callbacks_validation = [
            ModelCheckpoint(
                monitor="val_loss",
                save_top_k=1,
                filename="epoch-{epoch}-step-{step}-valloss-{val_loss:.8f}-mae-{metric_val/mae_epoch:.8f}",
                auto_insert_metric_name=False,
            ),
            WandbMetricsDefiner(),
        ]
        callbacks += callbacks_validation
    return callbacks


@dataclass
class NYUv2DepthParams:
    project: str
    checkpoint_path: str
    seed: int
    max_epochs: int

    dataset_dir: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    train_dataset_fraction: float

    num_subnetworks: int
    filter_base_count: int
    center_dropout_rate: float
    final_dropout_rate: float
    encoder_dropout_rate: float
    core_dropout_rate: float
    decoder_dropout_rate: float
    loss_buffer_size: int
    loss_buffer_temperature: float
    input_repetition_probability: float
    batch_repetitions: int
    loss: str
    weight_decay: float
    learning_rate: float

    @classmethod
    def from_namespace(cls, args: Namespace) -> 'NYUv2DepthParams':
        return cls(
            # training parameters
            project=args.project,
            checkpoint_path=args.checkpoint_path,
            seed=args.seed,
            max_epochs=args.max_epochs,

            # datamodule parameters
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            train_dataset_fraction=args.train_dataset_fraction,

            # model parameters
            num_subnetworks=args.num_subnetworks,
            filter_base_count=args.filter_base_count,
            center_dropout_rate=args.center_dropout_rate,
            final_dropout_rate=args.final_dropout_rate,
            encoder_dropout_rate=args.encoder_dropout_rate,
            core_dropout_rate=args.core_dropout_rate,
            decoder_dropout_rate=args.decoder_dropout_rate,
            loss_buffer_size=args.loss_buffer_size,
            loss_buffer_temperature=args.loss_buffer_temperature,
            input_repetition_probability=args.input_repetition_probability,
            batch_repetitions=args.batch_repetitions,
            loss=args.loss,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
        )


def main(params: NYUv2DepthParams):
    pl.seed_everything(params.seed)

    dm = NYUv2DepthDataModule.from_params(params)

    model = MimoUnetModel(
        in_channels=3,
        out_channels=2,
        num_subnetworks=params.num_subnetworks,
        filter_base_count=params.filter_base_count,
        center_dropout_rate=params.center_dropout_rate,
        final_dropout_rate=params.final_dropout_rate,
        encoder_dropout_rate=params.encoder_dropout_rate,
        core_dropout_rate=params.core_dropout_rate,
        decoder_dropout_rate=params.decoder_dropout_rate,
        loss_buffer_size=params.loss_buffer_size,
        loss_buffer_temperature=params.loss_buffer_temperature,
        input_repetition_probability=params.input_repetition_probability,
        batch_repetitions=params.batch_repetitions,
        loss=params.loss,
        weight_decay=params.weight_decay,
        learning_rate=params.learning_rate,
        seed=params.seed,
    )

    wandb_logger = WandbLogger(project=params.project, log_model=True, save_dir=params.checkpoint_path)
    wandb_logger.experiment.config.update(asdict(params))

    trainer = pl.Trainer(
        callbacks=default_callbacks(), 
        accelerator='gpu', 
        devices=1,
        precision=16,
        max_epochs=100,
        default_root_dir=params.checkpoint_path,
        log_every_n_steps=200,
        logger=wandb_logger,
    )

    trainer.started_at = str(datetime.now().isoformat(timespec="seconds"))
    trainer.fit(model, dm)
    wandb_logger.experiment.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument(
        "--project",
        type=str,
        default="MIMO NYUv2Depth",
        help="Specify the name of the project for wandb.",
    )
    parser.add_argument(
        "--checkpoint_path", 
        type=dir_path, 
        required=True,
        help="Path where the lightning logs and checkpoints should be saved to.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        required=True,
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Specify the maximum number of epochs to train.",
    )
    parser = NYUv2DepthDataModule.add_model_specific_args(parser)
    parser = MimoUnetModel.add_model_specific_args(parser)
    args = parser.parse_args()

    params = NYUv2DepthParams.from_namespace(args)
    logger.debug("command line arguments: %s", params)
    main(params)
