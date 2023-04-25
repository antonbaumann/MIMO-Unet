from argparse import ArgumentParser, Namespace
from datetime import datetime
import logging

import pytorch_lightning as pl
import torch
from losses import UncertaintyLoss
from models.unet import UNet
from ndvi_prediction.training import get_datamodule, get_default_callbacks, get_argument_parser, get_metrics_dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MimoUnetModel(pl.LightningModule):
    def __init__(
            self,
            nr_subnetworks: int,
            nr_input_channels: int,
            nr_output_channels: int,
            loss: str,
            weight_decay: float,
            learning_rate: float,
            seed: int,
        ):
        super().__init__()

        self.nr_subnetworks = nr_subnetworks
        self.nr_input_channels = nr_input_channels
        self.nr_output_channels = nr_output_channels
        self.criterion = UncertaintyLoss.from_name(loss)
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.seed = seed

        self.model = UNet(
            in_channels=nr_subnetworks * nr_input_channels,
            out_channels=nr_subnetworks * nr_output_channels * 2,
            bilinear=True,
        )

        metrics = ["MAE", "R2"]
        for mode in ["train", "val"]:
            for k, m in get_metrics_dict(mode=mode, metrics=metrics).items():
                setattr(self, "metric_" + k, m)

        self.save_hyperparameters()
        self.save_hyperparameters({"loss_name": loss})

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, S, C_in, H, W]
        Returns:
            p1: [B, S, C_out, H, W]
            p2: [B, S, C_out, H, W]
        """
        B, S, C_in, H, W = x.shape

        assert S == self.nr_subnetworks, "channel dimension must match nr_subnetworks"
        assert C_in == self.input_channels, "channel dimension must match input_channels"

        # reshape input tensor to match MIMO architecture
        # [B, S, C, H, W] -> [B, S*C, H, W]
        x = x.view(B, self.nr_subnetworks * self.nr_input_channels, H, W)

        # [B, 2*S*C_out, H, W]
        out = self.model(x)

        # [B, 2*S*C_out, H, W] -> [B, S, 2*C_out, H, W]
        out = out.view(B, self.nr_subnetworks, 2 * self.nr_output_channels, H, W)

        # [B, S, C_out, H, W]
        p1 = out[:, :, :self.nr_output_channels, ...]
        p2 = out[:, :, self.nr_output_channels:, ...]

        return p1, p2

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        p1, p2 = self(x)

        loss = self.loss_fn.forward(p1, p2, y)
        y_hat = self.loss_fn.mode(p1, p2)
        aleatoric_std = self.loss_fn.std(p1, p2)

        self.log("train_loss", loss, batch_size=self.trainer.datamodule.batch_size)
        self._log_metrics(y=y, y_hat=y_hat, mode="train", batch_idx=batch_idx)

        return {"loss": loss, "preds": y_hat, "std_map": aleatoric_std, "err_map": y_hat - y}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        p1, p2 = self(x)

        val_loss = self.loss_fn.forward(p1, p2, y)
        y_hat = self.loss_fn.mode(p1, p2)
        aleatoric_std = self.loss_fn.std(p1, p2)

        self.log("val_loss", val_loss, batch_size=self.trainer.datamodule.batch_size)

        self._log_metrics(y=y, y_hat=y_hat, mode="val", batch_idx=batch_idx)

        return {"loss": val_loss, "preds": y_hat, "std_map": aleatoric_std, "err_map": y_hat - y}
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        x = batch["image"]
        p1, p2 = self(x)
        y_hat = self.loss_fn.mode(p1, p2)
        aleatoric_std = self.loss_fn.std(p1, p2)
        return y_hat, aleatoric_std

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5, verbose=True)
        return [optimizer], [scheduler]
    
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group(title="NDVIModel")
        parser.add_argument("--nr_subnetworks", type=int, default=3)
        parser.add_argument("--nr_channels", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--loss", type=str, default="laplace_nll")
        parser.add_argument("--seed", type=int, default=42)

        parser = UNet.add_model_specific_args(parser)

        return parser


def main(args: Namespace):
    args_dict = vars(args)
    pl.seed_everything(args.seed)

    dm = get_datamodule(args_dict)

    epochs = args.max_epochs
    network_input_channels = len(dm.model_inputs)
    kwargs = {
        "gpus": 1,
        "max_epochs": epochs,
        "default_root_dir": args.checkpoint_path,
        "log_every_n_steps": 100,
    }
    additional_hparams = {
        "patch_size": str(dm.patch_size),
        "max_epochs": epochs,
        "model_inputs": str(dm.model_inputs),
        "normalization": "min_max",
        "dataset_size": len(dm),
        "dataset_path": args.dataset_dir,
        "batch_size": args.batch_size,
    }
    logger.debug("additional_hparams: %s", additional_hparams)
    model = MimoUnetModel(
        nr_subnetworks=args.nr_subnetworks,
        nr_input_channels=network_input_channels,
        nr_output_channels=1,
        loss=args.loss,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    callbacks = get_default_callbacks()
    trainer= pl.Trainer.from_argparse_args(args, callbacks=callbacks, **kwargs)
    print("Trainer:", trainer)
    trainer.started_at = str(datetime.now().isoformat(timespec="seconds"))
    trainer.fit(model, dm)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_argument_parser()
    parser = MimoUnetModel.add_model_specific_args(parser)
    args = parser.parse_args()
    logger.debug("command line arguments: %s", args)
    main(args)  