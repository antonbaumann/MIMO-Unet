from typing import List
from argparse import Namespace
from datetime import datetime
import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from utils import dir_path
from models.mimo_unet import MimoUnetModel
from ndvi_prediction.datamodule import get_datamodule, get_argument_parser
from ndvi_prediction.training_helpers import InputMonitor, OutputMonitor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def default_callbacks(validation: bool = True) -> List[pl.Callback]:
    callbacks = [
        # InputMonitor(),
        OutputMonitor(),
        ModelCheckpoint(save_last=True),
    ]
    if validation:
        callbacks_validation = [
            ModelCheckpoint(
                monitor="val_loss",
                save_top_k=5,
                filename="epoch-{epoch}-step-{step}-valloss-{val_loss:.8f}-mae-{metric_val/mae_epoch:.8f}",
                auto_insert_metric_name=False,
            ),
        ]
        callbacks += callbacks_validation
    return callbacks


def main(args: Namespace):
    pl.seed_everything(args.seed)

    dm = get_datamodule(args)

    model = MimoUnetModel(
        in_channels=len(dm.model_inputs),
        out_channels=len(dm.model_targets) * 2,
        num_subnetworks=args.num_subnetworks,
        filter_base_count=args.filter_base_count,
        center_dropout_rate=args.center_dropout_rate,
        final_dropout_rate=args.final_dropout_rate,
        input_repetition_probability=args.input_repetition_probability,
        batch_repetitions=args.batch_repetitions,
        loss=args.loss,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    wandb_logger = WandbLogger(project="MIMO Sen12TP")
    wandb_logger.watch(model, log="all", log_freq=500)
    wandb_logger.experiment.config.update(vars(args))

    trainer = pl.Trainer(
        callbacks=default_callbacks(), 
        accelerator='gpu', 
        devices=1,
        precision=32,
        max_epochs=args.max_epochs,
        default_root_dir=args.checkpoint_path,
        log_every_n_steps=300,
        logger=wandb_logger,
    )

    trainer.started_at = str(datetime.now().isoformat(timespec="seconds"))
    trainer.fit(model, dm)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_argument_parser()
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
    parser = MimoUnetModel.add_model_specific_args(parser)
    args = parser.parse_args()
    logger.debug("command line arguments: %s", args)
    main(args)  