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
from sen12tp.dataset import SEN12TP, Patchsize
import sen12tp.utils

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def make_predictions(model, dataset, device: str, batch_size: int = 5, num_workers=30):
    inputs = []
    y_preds = []
    y_trues = []
    aleatoric_vars = []
    epistemic_vars = []
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    for data in tqdm(loader):
        images = data['image'].to(device)
        y_pred, aleatoric_var, epistemic_var = model(images)

        y_pred = y_pred.cpu().detach()
        aleatoric_var = aleatoric_var.cpu().detach()
        epistemic_var = epistemic_var.cpu().detach()

        y_true = data['label'].cpu().detach()

        inputs.append(images.cpu().detach())
        y_preds.append(y_pred)
        y_trues.append(y_true)
        aleatoric_vars.append(aleatoric_var)
        epistemic_vars.append(epistemic_var)

    inputs = torch.cat(inputs, dim=0)
    y_preds = torch.cat(y_preds, dim=0).clip(min=0, max=1)
    y_trues = torch.cat(y_trues, dim=0).clip(min=0, max=1)
    aleatoric_vars = torch.cat(aleatoric_vars, dim=0)
    epistemic_vars = torch.cat(epistemic_vars, dim=0)
    
    return (
        inputs,
        y_preds[:, 0], 
        y_trues[:, 0], 
        aleatoric_vars[:, 0], 
        epistemic_vars[:, 0],
        aleatoric_vars[:, 0] + epistemic_vars[:, 0],
    )


def convert_to_pandas(y_preds, y_trues, aleatoric_vars, epistemic_vars, combined_vars):
    data = torch.stack([
        y_preds.flatten(),
        y_trues.flatten(),
        torch.abs(y_trues.flatten() - y_preds.flatten()),
        aleatoric_vars.sqrt().flatten(),
        epistemic_vars.sqrt().flatten(),  
        combined_vars.sqrt().flatten(),  
    ], axis=0).T

    df = pd.DataFrame(
        data=data.cpu().numpy(),
        columns=['y_pred', 'y_true', 'error', 'aleatoric_std', 'epistemic_std', 'combined_std']
    )
    return df


def create_precision_recall_plot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by='combined_std', ascending=False)
    
    percentiles = np.arange(100)/100.
    cutoff_inds = (percentiles * df.shape[0]).astype(int)
    
    mae = [df.iloc[cutoff:]["error"].mean() for cutoff in tqdm(cutoff_inds)]
    mse = [np.square(df.iloc[cutoff:]["error"]).mean() for cutoff in tqdm(cutoff_inds)]
    
    df_cutoff = pd.DataFrame({'percentile': percentiles, 'mae': mae, 'rmse': np.sqrt(mse)})
    
    return df_cutoff


def compute_ppf(params):
    p, y_pred, aleatoric_std, distribution = params
    return distribution.ppf(p, loc=y_pred, scale=aleatoric_std / np.sqrt(2))
    
def create_calibration_plot(df: pd.DataFrame, distribution, processes, num_samples: int) -> pd.DataFrame:
    df = df.sample(n=num_samples, replace=False)
    y_true = df['y_true'].to_numpy()
    y_pred = df['y_pred'].to_numpy()
    aleatoric_std = df['aleatoric_std'].to_numpy()

    expected_p = np.arange(41) / 40.

    print('- computing ppfs')
    with mp.Pool(processes=processes) as pool:
        params = [(p, y_pred, aleatoric_std, distribution) for p in expected_p]
        results = pool.imap(compute_ppf, params, chunksize=1)
        ppfs = np.array(list(tqdm(results, total=len(expected_p))))

    print('- computing observed_p')
    below = y_true[None, :] < ppfs
    observed_p = below.mean(axis=1)
    
    df_calibration = pd.DataFrame({'Expected Conf.': expected_p, 'Observed Conf.': observed_p})
    return df_calibration


def main(
    dataset_path: str,
    model_checkpoint_paths: List[str],
    monte_carlo_steps: int,
    result_dir: str,
    device: str,
    processes: int = None,
    batch_size: int = 5,
    patch_size: int = 256,
    stride: int = 249,
) -> None:
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=False)

    model = EnsembleModule(
        checkpoint_paths=model_checkpoint_paths,
        monte_carlo_steps=monte_carlo_steps,
        return_raw_predictions=False,
    )
    model.to(device)

    dataset = SEN12TP(
        path=dataset_path,
        patch_size=Patchsize(patch_size, patch_size),
        stride=stride,
        model_inputs=['VV_sigma0', 'VH_sigma0'],
        model_targets=['NDVI'],
        transform=sen12tp.utils.min_max_transform,
        clip_transform=sen12tp.utils.default_clipping_transform,
    )

    print(f"Making predictions ...")
    inputs, y_preds, y_trues, aleatoric_vars, epistemic_vars, combined_vars = make_predictions(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        device=device,
    )

    print(f"Saving predictions ...")
    np.save(result_dir / f"inputs.npy", inputs.numpy())
    np.save(result_dir / f"y_preds.npy", y_preds.numpy())
    np.save(result_dir / f"y_trues.npy", y_trues.numpy())
    np.save(result_dir / f"aleatoric_vars.npy", aleatoric_vars.numpy())
    np.save(result_dir / f"epistemic_vars.npy", epistemic_vars.numpy())
    
    print(f"Computing metrics ...")
    df = convert_to_pandas(
        y_preds=y_preds,
        y_trues=y_trues,
        aleatoric_vars=aleatoric_vars,
        epistemic_vars=epistemic_vars,
        combined_vars=combined_vars,
    )

    print(f"Saving dataframes ...")
    df.to_pickle(result_dir / f"df_pixels.pkl")
    
    print(f"Creating data for precision-recall plot ...")
    df_cutoff = create_precision_recall_plot(df)
    df_cutoff.to_csv(result_dir / f'precision_recall.csv', index=False)
    
    print(f"Creating data for calibration plot...")
    processes = min(mp.cpu_count(), processes) 
    df_calibration = create_calibration_plot(df, scipy.stats.norm, processes=processes, num_samples=100_000_000)
    df_calibration.to_csv(result_dir / f"calibration.csv", index=False)

    print(f"Finished processing dataset!")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint_paths", nargs="+", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--monte_carlo_steps", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=249)

    args = parser.parse_args()

    main(
        dataset_path=args.dataset_dir,
        model_checkpoint_paths=args.model_checkpoint_paths,
        monte_carlo_steps=args.monte_carlo_steps,
        result_dir=args.result_dir,
        device=args.device,
        processes=args.processes,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        stride=args.stride,
    )
