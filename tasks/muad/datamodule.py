from typing import Optional, Tuple
from argparse import Namespace, ArgumentParser
import pytorch_lightning as pl
import torch
import os

from utils import dir_path, parse_image_dimensions
from datasets.muad import MUADDepthDataset

class MUADDepthDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        normalize: bool,
        dsize: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize = normalize
        self.dsize = dsize

    def setup(
        self,
        stage: Optional[str] = None,
    ) -> None:

        self.data_train = MUADDepthDataset(
            dataset_path=os.path.join(self.dataset_dir, "train"),
            dsize=self.dsize,
            normalize=self.normalize,
            shuffle_on_load=False,
        )

        self.data_val = MUADDepthDataset(
            dataset_path=os.path.join(self.dataset_dir, "val"),
            dsize=self.dsize,
            normalize=self.normalize,
            shuffle_on_load=True,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=self.pin_memory,
        )
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.data_valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.data_valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
        )
    
def get_datamodule(args: Namespace) -> MUADDepthDatamodule:
    return MUADDepthDatamodule(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

def add_datamodule_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = parent_parser.add_argument_group(title="NYUv2DepthDataModule")

    parser.add_argument(
        "--dataset_dir",
        type=dir_path,
        default="/scratch/datasets/sen12tp-cropland-splitted/",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Specify the batch size.",
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Specify the number of workers.",
    )

    parser.add_argument(
        "--pin_memory",
        type=bool,
        default=True,
        help="Specify whether to pin memory.",
    )

    parser.add_argument(
        "--dsize",
        nargs='+', 
        type=parse_image_dimensions,
    )

    parser.add_argument(
        "--normalize",
        type=bool,
        default=True,
    )

    return parent_parser