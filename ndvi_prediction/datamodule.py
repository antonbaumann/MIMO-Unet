from argparse import ArgumentParser, Namespace
import logging

from sen12tp.datamodule import SEN12TPDataModule
from sen12tp.dataset import Patchsize
import sen12tp.utils

from utils import dir_path


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_datamodule(args: Namespace) -> SEN12TPDataModule:
    dm = SEN12TPDataModule(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        patch_size=Patchsize(
            args.patch_size, 
            args.patch_size,
        ),
        stride=args.stride,
        model_inputs=args.input,
        model_targets=args.target,
        transform=sen12tp.utils.min_max_transform,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle_train=True,
        drop_last_train=True,
    )
    dm.setup()
    return dm


def get_argument_parser() -> ArgumentParser:
    parent_parser = ArgumentParser()
    parser = parent_parser.add_argument_group(title="Sen12tpDataModule")

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
        "--patch_size",
        type=int,
        default=256,
        help="Specify the patch size.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=249,
        help="Specify the stride.",
    )

    parser.add_argument(
        "-i",
        "--input",
        action="append",
        required=True,
        help=f"Set the used model inputs.",
    )

    parser.add_argument(
        "-t",
        "--target",
        action="append",
        required=True,
        help="Specify the targets the model should predict.",
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Specify the number of workers.",
    )

    return parent_parser