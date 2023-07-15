from typing import Optional
from argparse import ArgumentParser
from dataclasses import dataclass
import pytorch_lightning as pl
import torch
import os

from mimo.utils import dir_path
from mimo.datasets.make3d import Make3dDepthDataset


class Make3dDepthDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        normalize: bool = True,
        train_dataset_fraction: float = 1.0,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize = normalize
        self.train_dataset_fraction = train_dataset_fraction

    def setup(
        self,
        stage: Optional[str] = None,
    ) -> None:
        self.data_train = Make3dDepthDataset(
            dataset_path=os.path.join(self.dataset_dir, 'train'),
            normalize=self.normalize,
            shuffle_on_load=False,
            use_fraction=self.train_dataset_fraction,
        )

        self.data_valid = Make3dDepthDataset(
            dataset_path=os.path.join(self.dataset_dir, 'train'),
            normalize=self.normalize,
            shuffle_on_load=True,
        )

        self.data_test = Make3dDepthDataset(
            dataset_path=os.path.join(self.dataset_dir, 'test'),
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
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
        )
    
    @classmethod
    def from_params(cls, params: dataclass) -> "Make3dDepthDataModule":
        return cls(
            dataset_dir=params.dataset_dir,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            pin_memory=params.pin_memory,
            train_dataset_fraction=params.train_dataset_fraction,
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group(title="Make3dDepthDataModule")

        parser.add_argument(
            "--dataset_dir",
            type=dir_path,
            required=True,
            help="Specify the dataset directory.",
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
            "--train_dataset_fraction",
            type=float,
            default=1.0,
            help="Specify the fraction of the training dataset to use.",
        )

        return parent_parser
