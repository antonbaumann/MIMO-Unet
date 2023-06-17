from argparse import ArgumentParser
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.stats
import numpy as np
import pandas as pd
from pathlib import Path
import os
import multiprocessing as mp

from models.ensemble import EnsembleModule
from datasets.nyuv2 import NYUv2DepthDataset
from datasets.make3d import Make3dDepthDataset

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def make_predictions(model, dataset, device: str, batch_size: int = 32, epsilon: float = 0.0):
    inputs = []
    y_preds = []
    y_trues = []
    log_params = []
    
    loader = DataLoader(dataset, batch_size=batch_size)

    for data in tqdm(loader):
        images = data['image'].to(device)
        labels = data['label'].to(device)

        images.requires_grad = True

        y_pred, log_param = model(images)

        loss = model.loss_fn(y_pred, labels, log_param)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = images.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(images, epsilon, data_grad)

        # Predict on the perturbed image
        y_pred, log_param = model(perturbed_data)

        y_pred = y_pred.cpu().detach()
        log_param = log_param.cpu().detach()
        y_true = labels.cpu().detach()

        inputs.append(perturbed_data.cpu().detach())
        y_preds.append(y_pred)
        y_trues.append(y_true)
        log_params.append(log_param)

    inputs = torch.cat(inputs, dim=0)
    y_preds = torch.cat(y_preds, dim=0).clip(min=0, max=1)
    y_trues = torch.cat(y_trues, dim=0).clip(min=0, max=1)
    log_params = torch.cat(log_params, dim=0)
    
    aleatoric_var, epistemic_var = compute_uncertainties(
        model.loss_fn,
        y_preds=y_preds,
        log_params=log_params,
    )
    
    return (
        inputs,
        y_preds.mean(axis=1)[:, 0], 
        y_trues[:, 0], 
        aleatoric_var[:, 0], 
        epistemic_var[:, 0],
        aleatoric_var[:, 0] + epistemic_var[:, 0],
    )


def convert_to_pandas(y_preds, y_trues, aleatoric_vars, epistemic_vars, combined_vars):
    data = np.stack([
        y_preds.numpy().flatten(),
        y_trues.numpy().flatten(), 
        np.sqrt(aleatoric_vars.numpy()).flatten(),
        np.sqrt(epistemic_vars.numpy()).flatten(),  
        np.sqrt(combined_vars.numpy()).flatten(),  
    ], axis=0).T
    
    df = pd.DataFrame(
        data=data,
        columns=['y_pred', 'y_true', 'aleatoric_std', 'epistemic_std', 'combined_std']
    )
    return df


def compute_uncertainties(criterion, y_preds, log_params):
    """
    Args:
        y_preds: [B, S, C, H, W]
    """
    _, S, _, _, _ = y_preds.shape
    
    stds = criterion.std(y_preds, log_params)
    aleatoric_variance = torch.square(stds).mean(dim=1)
    
    if S > 1:
        y_preds_mean = y_preds.mean(dim=1, keepdims=True)
        epistemic_variance = torch.square(y_preds - y_preds_mean).sum(dim=1) / (S - 1)
    else:
        epistemic_variance = torch.zeros_like(aleatoric_variance)
        
    return aleatoric_variance, epistemic_variance


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df['error'] = np.abs(df['y_pred'] - df['y_true'])
    df['entropy'] = 0.5 * np.log(2 * np.pi * np.exp(1.) * (df['combined_std'] ** 2))
    return df

def main(
    model_checkpoint_paths: List[str],
    monte_carlo_steps: int,
    datasets: List[Tuple[str, str]],
    result_dir: str,
    device: str,
) -> None:
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=False)

    model = EnsembleModule(
        checkpoint_paths=model_checkpoint_paths,
        monte_carlo_steps=monte_carlo_steps,
    )
    model.to(device)

    for dataset_name, dataset_path in datasets:
        for noise_level in [0, 0.01, 0.02]:
            if dataset_name == 'nyuv2depth':
                dataset = NYUv2DepthDataset(
                    dataset_path=dataset_path,
                    normalize=True,
                )
            elif dataset_name == 'make3d':
                dataset = Make3dDepthDataset(
                    dataset_path=dataset_path,
                    normalize=True,
                )
            else:
                raise ValueError(f"Unknown dataset `{dataset_name}`!")

            print(f"Making predictions on {dataset_name}...")
            inputs, y_preds, y_trues, aleatoric_vars, epistemic_vars, combined_vars = make_predictions(
                model=model,
                dataset=dataset,
                batch_size=32,
                device=device,
                epsilon=noise_level,
            )

            print(f"Saving predictions on {dataset_name}...")
            np.save(result_dir / f"{dataset_name}_{noise_level}_inputs.npy", inputs.numpy())
            np.save(result_dir / f"{dataset_name}_{noise_level}_y_preds.npy", y_preds.numpy())
            np.save(result_dir / f"{dataset_name}_{noise_level}_y_trues.npy", y_trues.numpy())
            np.save(result_dir / f"{dataset_name}_{noise_level}_aleatoric_vars.npy", aleatoric_vars.numpy())
            np.save(result_dir / f"{dataset_name}_{noise_level}_epistemic_vars.npy", epistemic_vars.numpy())
            
            print(f"Computing metrics on {dataset_name}...")
            df = convert_to_pandas(
                y_preds=y_preds,
                y_trues=y_trues,
                aleatoric_vars=aleatoric_vars,
                epistemic_vars=epistemic_vars,
                combined_vars=combined_vars,
            )
            df = compute_metrics(df)

            print(f"Saving dataframes for {dataset_name}...")
            df.to_pickle(result_dir / f"{dataset_name}_{noise_level}_metrics.pkl")

            print(f"Finished processing dataset `{dataset_name}`!")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint_paths", nargs="+", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--nyuv2_dataset_dir", type=str, required=True)
    parser.add_argument("--make3d_dataset_dir", type=str, required=True)
    parser.add_argument("--monte_carlo_steps", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    main(
        model_checkpoint_paths=args.model_checkpoint_paths,
        monte_carlo_steps=args.monte_carlo_steps,
        datasets=[
            ("make3d", os.path.join(args.make3d_dataset_dir, "test")),
            ("nyuv2depth", os.path.join(args.nyuv2_dataset_dir, "depth_test.h5")),
        ],
        result_dir=args.result_dir,
        device=args.device,
    )
