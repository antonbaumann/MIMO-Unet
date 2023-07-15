from typing import List
from argparse import Namespace, ArgumentParser
from datetime import datetime
import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from mimo.utils import dir_path
from mimo.models.evidential_unet import EvidentialUnetModel
from mimo.tasks.sen12tp.sen12tp_datamodule import get_datamodule, add_datamodule_args
from mimo.tasks.sen12tp.callbacks import OutputMonitor

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
                save_top_k=2,
                filename="epoch-{epoch}-step-{step}-valloss-{val_loss:.8f}-mae-{metric_val/mae_epoch:.8f}",
                auto_insert_metric_name=False,
            ),
        ]
        callbacks += callbacks_validation
    return callbacks


def main(args: Namespace):
    pl.seed_everything(args.seed)

    dm = get_datamodule(args)

    model = EvidentialUnetModel(
        in_channels=len(dm.model_inputs),
        out_channels=4,
        filter_base_count=args.filter_base_count,
        center_dropout_rate=args.center_dropout_rate,
        final_dropout_rate=args.final_dropout_rate,
        encoder_dropout_rate=args.encoder_dropout_rate,
        core_dropout_rate=args.core_dropout_rate,
        decoder_dropout_rate=args.decoder_dropout_rate,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    wandb_logger = WandbLogger(project=args.project)
    wandb_logger.experiment.config.update(vars(args))

    trainer = pl.Trainer(
        callbacks=default_callbacks(), 
        accelerator='gpu', 
        devices=1,
        precision=16,
        max_epochs=args.max_epochs,
        default_root_dir=args.checkpoint_path,
        log_every_n_steps=300,
        logger=wandb_logger,
    )

    trainer.started_at = str(datetime.now().isoformat(timespec="seconds"))
    trainer.fit(model, dm)
    wandb_logger.experiment.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()
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
        default=40,
        help="Specify the maximum number of epochs to train.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="MIMO Sen12TP",
        help="Specify the name of the wandb project.",
    )
    parser = add_datamodule_args(parser)
    parser = EvidentialUnetModel.add_model_specific_args(parser)
    args = parser.parse_args()
    logger.debug("command line arguments: %s", args)
    main(args)  