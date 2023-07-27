from typing import List
from argparse import Namespace, ArgumentParser
from datetime import datetime
import logging

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

from mimo.utils import dir_path
from mimo.models.mimo_unet import MimoUnetModel
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

    model = MimoUnetModel(
        in_channels=len(dm.model_inputs),
        out_channels=len(dm.model_targets) * args.num_loss_function_params,
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
        seed=args.seed,
    )

    model.compile()

    wandb_logger = WandbLogger(project=args.project)
    wandb_logger.experiment.config.update(vars(args))

    trainer = pl.Trainer(
        callbacks=default_callbacks(), 
        accelerator='gpu', 
        devices=1,
        precision="16-mixed",
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
    parser.add_argument(
        "--num_loss_function_params",
        type=int,
        default=2,
        help="Specify the number of parameters of the loss function.",
    )
    parser = add_datamodule_args(parser)
    parser = MimoUnetModel.add_model_specific_args(parser)
    args = parser.parse_args()
    logger.debug("command line arguments: %s", args)
    main(args)  