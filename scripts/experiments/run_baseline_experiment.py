from argparse import ArgumentParser
from typing import Any
from tqdm import tqdm

import scripts.train.train_nyuv2_depth as train_nyuv2_depth
from utils import dir_path


class ExperimentConfigurator:
    def __init__(
        self,
        project: str,
        checkpoint_path: str,
        max_epochs: int,
        batch_size: int,
        dataset_dir: str,
        num_workers: int,
    ):
        self.project = project
        self.checkpoint_path = checkpoint_path
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.num_workers = num_workers

    def make_config(
        self,
        seed: int,
        num_subnetworks: int,
        filter_base_count: int,
        encoder_dropout_rate: float,
        core_dropout_rate: float,
        decoder_dropout_rate: float,
        loss_buffer_size: float,
        loss_buffer_temperature: float,
        input_repetition_probability: float,
    ) -> train_nyuv2_depth.NYUv2DepthParams:
        return train_nyuv2_depth.NYUv2DepthParams(
            project=self.project,
            checkpoint_path=self.checkpoint_path,
            seed=seed,
            max_epochs=self.max_epochs,
            dataset_dir=self.dataset_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            num_subnetworks=num_subnetworks,
            filter_base_count=filter_base_count,
            center_dropout_rate=0.0,
            final_dropout_rate=0.0,
            encoder_dropout_rate=encoder_dropout_rate,
            core_dropout_rate=core_dropout_rate,
            decoder_dropout_rate=decoder_dropout_rate,
            loss_buffer_size=loss_buffer_size,
            loss_buffer_temperature=loss_buffer_temperature,
            input_repetition_probability=input_repetition_probability,
            batch_repetitions=1,
            loss='laplace_nll',
            weight_decay=0.0,
            learning_rate=1e-4,
        )


def run(
    project: str,
    checkpoint_path: str,
    dataset_dir: str,
    num_workers: int,
):
    c = ExperimentConfigurator(
        project=project,
        checkpoint_path=checkpoint_path,
        max_epochs=100,
        dataset_dir=dataset_dir,
        num_workers=num_workers,
    )

    configs = []

    for seed in [1, 2, 3]:

        # mimo models (2 subnetworks)
        for filter_base_count in [30, 45]:
            for lb_size, lb_temp in [(0, 1), (10, 0.3)]:
                for input_repetition_probability in [0.0, 0.2, 0.6]:
                    configs.append(c.make_config(
                        seed=seed,
                        num_subnetworks=2,
                        filter_base_count=filter_base_count,
                        encoder_dropout_rate=0,
                        core_dropout_rate=0,
                        decoder_dropout_rate=0,
                        loss_buffer_size=lb_size,
                        loss_buffer_temperature=lb_temp,
                        input_repetition_probability=input_repetition_probability,
                    ))

        # mimo models (3 subnetworks)
        for filter_base_count in [30]:
            for lb_size, lb_temp in [(0, 1), (10, 0.3)]:
                for input_repetition_probability in [0.0, 0.2, 0.6]:
                    configs.append(c.make_config(
                        seed=seed,
                        num_subnetworks=3,
                        filter_base_count=filter_base_count,
                        encoder_dropout_rate=0,
                        core_dropout_rate=0,
                        decoder_dropout_rate=0,
                        loss_buffer_size=lb_size,
                        loss_buffer_temperature=lb_temp,
                        input_repetition_probability=input_repetition_probability,
                    ))

        # mc dropout
        for filter_base_count in [30, 60]:
            configs.append(c.make_config(
                seed=seed,
                num_subnetworks=1,
                filter_base_count=filter_base_count,
                encoder_dropout_rate=0,
                core_dropout_rate=0.3,
                decoder_dropout_rate=0,
                loss_buffer_size=0,
                loss_buffer_temperature=1,
                input_repetition_probability=0.0,
            ))

        # ensemble
        for filter_base_count in [30, 60]:
            configs.append(c.make_config(
                seed=seed,
                num_subnetworks=1,
                filter_base_count=filter_base_count,
                encoder_dropout_rate=0,
                core_dropout_rate=0,
                decoder_dropout_rate=0,
                loss_buffer_size=0,
                loss_buffer_temperature=1,
                input_repetition_probability=0.0,
            ))

    print(f"Running {len(configs)} experiments.")

    for config in tqdm(configs):
        print(f"Running experiment with config: {config}")
        train_nyuv2_depth.main(config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--project", 
        type=str, 
        default="MIMO NYUv2Depth Baselines", 
        help="Specify the name of the project for wandb.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=dir_path,
        required=True,
        help="Path where the lightning logs and checkpoints should be saved to.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=dir_path,
        required=True,
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Specify the number of workers for the dataloader.",
    )
    args = parser.parse_args()
    run(
        args.project, 
        args.checkpoint_path, 
        args.dataset_dir,
        args.num_workers,
    )
