from argparse import Namespace
from datetime import datetime
import logging

import pytorch_lightning as pl
from models.mimo_unet import MimoUnetModel
from ndvi_prediction.training import get_datamodule, get_default_callbacks, get_argument_parser

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(args: Namespace):
    args_dict = vars(args)
    pl.seed_everything(args.seed)

    dm = get_datamodule(args_dict)

    model = MimoUnetModel(
        in_channels=len(dm.model_inputs),
        out_channels=len(dm.model_targets) * 2,
        num_subnetworks=args.num_subnetworks,
        filter_base_count=args.filter_base_count,
        center_dropout_rate=args.center_dropout_rate,
        final_dropout_rate=args.final_dropout_rate,
        loss=args.loss,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    trainer = pl.Trainer.from_argparse_args(
        args, 
        callbacks=get_default_callbacks(), 
        accelerator='gpu', 
        devices=1,
        precision=32,
        max_epochs=args.max_epochs,
        default_root_dir=args.checkpoint_path,
        log_every_n_steps=300,
        # limit_train_batches=150,
    )
    trainer.started_at = str(datetime.now().isoformat(timespec="seconds"))
    trainer.fit(model, dm)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_argument_parser()
    parser.add_argument("--seed", type=int, required=True)
    parser = MimoUnetModel.add_model_specific_args(parser)
    args = parser.parse_args()
    logger.debug("command line arguments: %s", args)
    main(args)  