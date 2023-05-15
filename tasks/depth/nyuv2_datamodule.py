from typing import Optional
from argparse import ArgumentParser, Namespace
import pytorch_lightning as pl
import torch
import h5py
import os

from utils import dir_path
from datasets.nyuv2 import NYUv2DepthDataset


class NYUv2DepthDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize = normalize

    def setup(
        self,
        stage: Optional[str] = None,
    ) -> None:
        h5_train = h5py.File(os.path.join(self.dataset_dir, "depth_train.h5"), "r")
        h5_test = h5py.File(os.path.join(self.dataset_dir, "depth_test.h5"), "r")

        self.data_train = NYUv2DepthDataset(
            dict(image=h5_train["image"], label=h5_train["depth"]),
            normalize=self.normalize,
            shuffle_on_load=False,
        )

        self.data_test = NYUv2DepthDataset(
            dict(image=h5_test["image"], label=h5_test["depth"]),
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
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
        )


def get_datamodule(args: Namespace) -> NYUv2DepthDataModule:
    return NYUv2DepthDataModule(
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

    return parent_parser
