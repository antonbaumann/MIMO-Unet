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

from models.ensemble import EnsembleModule
from datasets.nyuv2 import NYUv2DepthDataset


def make_predictions(model, dataset, device: str, batch_size: int = 32):
    y_preds = []
    y_trues = []
    log_params = []
    
    loader = DataLoader(dataset, batch_size=batch_size)

    for data in tqdm(loader):
        images = data['image'].to(device)
        y_pred, log_param = model(images)

        y_pred = y_pred.cpu().detach()
        log_param = log_param.cpu().detach()
        y_true = data['label'].cpu().detach()

        y_preds.append(y_pred)
        y_trues.append(y_true)
        log_params.append(log_param)

    y_preds = torch.cat(y_preds, dim=0).clip(min=0, max=1)
    y_trues = torch.cat(y_trues, dim=0).clip(min=0, max=1)
    log_params = torch.cat(log_params, dim=0)
    
    aleatoric_var, epistemic_var = compute_uncertainties(
        model.loss_fn,
        y_preds=y_preds,
        log_params=log_params,
    )
    
    return (
        y_preds.mean(axis=1)[:, 0], 
        y_trues[:, 0], 
        aleatoric_var, 
        epistemic_var,
        aleatoric_var + epistemic_var,
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


def create_precision_recall_plot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by='combined_std', ascending=False)
    
    percentiles = np.arange(100)/100.
    cutoff_inds = (percentiles * df.shape[0]).astype(int)
    
    mae = [df.iloc[cutoff:]["error"].mean() for cutoff in cutoff_inds]
    mse = [np.square(df.iloc[cutoff:]["error"]).mean() for cutoff in cutoff_inds]
    
    df_cutoff = pd.DataFrame({'percentile': percentiles, 'mae': mae, 'rmse': np.sqrt(mse)})
    
    return df_cutoff


def create_calibration_plot(df: pd.DataFrame, distribution) -> pd.DataFrame:
    expected_p = np.arange(41)/40.
    observed_p = []
    for p in tqdm(expected_p):
        ppf = distribution.ppf(p, loc=df['y_pred'], scale=df['aleatoric_std'] / np.sqrt(2))
        obs_p = (df['y_true'] < ppf).mean()
        observed_p.append(obs_p)
    df_calibration = pd.DataFrame({'Expected Conf.': expected_p, 'Observed Conf.': observed_p})
    return df_calibration


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
        dataset = NYUv2DepthDataset(
            dataset_path=dataset_path,
            normalize=True,
        )

        print(f"Making predictions on {dataset_name}...")
        y_preds, y_trues, aleatoric_vars, epistemic_vars, combined_vars = make_predictions(
            model=model,
            dataset=dataset,
            batch_size=32,
            device=device,
        )

        print(f"Saving predictions on {dataset_name}...")
        np.save(result_dir / f"{dataset_name}_y_preds.npy", y_preds.numpy())
        np.save(result_dir / f"{dataset_name}_y_trues.npy", y_trues.numpy())
        np.save(result_dir / f"{dataset_name}_aleatoric_vars.npy", aleatoric_vars.numpy())
        np.save(result_dir / f"{dataset_name}_epistemic_vars.npy", epistemic_vars.numpy())
        
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
        df.to_pickle(result_dir / f"{dataset_name}_metrics.pkl.gz")
        
        df_cutoff = create_precision_recall_plot(df)
        df_cutoff.to_csv(result_dir / f"{dataset_name}_precision_recall.csv", index=False)
        
        df_calibration = create_calibration_plot(df, scipy.stats.norm)
        df_calibration.to_csv(result_dir / f"{dataset_name}_calibration.csv", index=False)

        print(f"Finished processing dataset `{dataset_name}`!")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint_paths", nargs="+", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--monte_carlo_steps", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    main(
        model_checkpoint_paths=args.model_checkpoint_paths,
        monte_carlo_steps=args.monte_carlo_steps,
        datasets=[
            ("test", os.path.join(args.dataset_dir, "depth_test.h5")),
            ("ood", os.path.join(args.dataset_dir, "apolloscape_test.h5")),
        ],
        result_dir=args.result_dir,
        device=args.device,
    )
