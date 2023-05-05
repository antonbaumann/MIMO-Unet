from argparse import ArgumentParser, Namespace
from datetime import datetime
import logging

import pytorch_lightning as pl
import torch
from losses import UncertaintyLoss
from mimo_unet.model import MimoUNet
from ndvi_prediction.training import get_datamodule, get_default_callbacks, get_argument_parser, get_metrics_dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MimoUnetModel(pl.LightningModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_subnetworks: int,
            filter_base_count: int,
            center_dropout_rate: float,
            final_dropout_rate: float,
            loss: str,
            weight_decay: float,
            learning_rate: float,
            seed: int,
        ):
        super().__init__()

        # model parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subnetworks = num_subnetworks
        self.filter_base_count = filter_base_count
        self.center_dropout_rate = center_dropout_rate
        self.final_dropout_rate = final_dropout_rate

        # training parameters
        self.loss_fn = UncertaintyLoss.from_name(loss)
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.seed = seed

        self.ndvi_kwargs = {"vmin": 0, "vmax": 1, "cmap": "RdYlBu"}

        self.model = MimoUNet( 
            in_channels=in_channels,
            out_channels=out_channels,
            num_subnetworks=num_subnetworks,
            filter_base_count=filter_base_count,
            center_dropout_rate=center_dropout_rate,
            final_dropout_rate=final_dropout_rate,
            bilinear=True,
            use_pooling_indices=False,
        )

        self.save_hyperparameters()
        self.save_hyperparameters({"loss_name": loss})

    def _reshape_for_subnetwors(self, x: torch.Tensor, repeat: bool = False):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            x: [B // S, S, C, H, W]
        """
        B, C, H, W = x.shape
        assert B % self.num_subnetworks == 0, "batch dimension must be divisible by num_subnetworks"

        # [B, C, H, W] -> [B // S, S, C, H, W]
        if not repeat:
            return x.view(B // self.num_subnetworks, self.num_subnetworks, C, H, W)
        else:
            x = x[:, None, :, :, :]
            return x.repeat(1, self.num_subnetworks, 1, 1, 1)

    def _reshape_for_plotting(self, x: torch.Tensor):
        """
        Args:
            x: [B // S, S, C, H, W]
        Returns:
            x: [B, C, H, W]
        """
        B_, S, C, H, W = x.shape
        # [B // S, S, C, H, W] -> [B, C, H, W]
        return x.view(B_ * S, C, H, W)

    def _log_metrics(self, y_hat, y, mode: str, batch_idx: int) -> None:
        if mode != "val" and (batch_idx + 1) % self.trainer.log_every_n_steps == 0:
            return

        # global metrics
        metric_filter = lambda attr: attr.startswith("metric_") and attr.count("/") == 1 and mode in attr
        metric_name_list = filter(metric_filter, dir(self))

        for metric_name in metric_name_list:
            metric = getattr(self, metric_name)
            metric(y_hat.detach().flatten(), y.detach().flatten())
            self.log(
                metric_name,
                metric,
                on_step=True,
                on_epoch=True,
                metric_attribute=metric_name,
                batch_size=self.trainer.datamodule.batch_size,
            )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, S, C_in, H, W]
        Returns:
            p1: [B, S, C_out, H, W]
            p2: [B, S, C_out, H, W]
        """
        B, S, C_in, H, W = x.shape

        assert S == self.num_subnetworks, "subnetwork dimension must match num_subnetworks"
        assert C_in == self.in_channels, "channel dimension must match in_channels"

        # [B, S, 2*C_out, H, W]
        out = self.model(x)

        # [B, S, C_out, H, W]
        p1 = out[:, :, :self.out_channels // 2, ...]
        p2 = out[:, :, self.out_channels // 2:, ...]

        return p1, p2

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        
        x = self._reshape_for_subnetwors(x)
        y = self._reshape_for_subnetwors(y)

        p1, p2 = self(x)

        loss = self.loss_fn.forward(p1, p2, y)
        y_hat = self.loss_fn.mode(p1, p2)
        aleatoric_std = self.loss_fn.std(p1, p2)

        self.log("train_loss", loss, batch_size=self.trainer.datamodule.batch_size)
        self._log_metrics(y=y, y_hat=y_hat, mode="train", batch_idx=batch_idx)

        return {
            "loss": loss,
            "preds": self._reshape_for_plotting(y_hat), 
            "std_map": self._reshape_for_plotting(aleatoric_std), 
            "err_map": self._reshape_for_plotting(y_hat - y),
        }
    
    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]

        x = self._reshape_for_subnetwors(x, repeat=True)
        y = self._reshape_for_subnetwors(y, repeat=True)

        p1, p2 = self(x)

        val_loss = self.loss_fn.forward(p1, p2, y)
        y_hat = self.loss_fn.mode(p1, p2)
        aleatoric_std = self.loss_fn.std(p1, p2)

        self.log("val_loss", val_loss, batch_size=self.trainer.datamodule.batch_size)

        self._log_metrics(y=y, y_hat=y_hat, mode="val", batch_idx=batch_idx)

        return {
            "loss": val_loss, 
            "preds": self._reshape_for_plotting(y_hat), 
            "std_map": self._reshape_for_plotting(aleatoric_std), 
            "err_map": self._reshape_for_plotting(y_hat - y),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5, verbose=True)
        return [optimizer], [scheduler]
    
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group(title="NDVIModel")
        parser.add_argument("--num_subnetworks", type=int, default=3)
        parser.add_argument("--filter_base_count", type=int, default=32)
        parser.add_argument("--center_dropout_rate", type=float, default=0)
        parser.add_argument("--final_dropout_rate", type=float, default=0)

        parser.add_argument("--loss", type=str, default="laplace_nll")
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=0)
        return parent_parser


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
        gpus=1,
        max_epochs=args.max_epochs,
        default_root_dir=args.checkpoint_path,
        log_every_n_steps=100,
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