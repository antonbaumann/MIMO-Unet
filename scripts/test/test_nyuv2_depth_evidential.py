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

from models.evidential_unet import EvidentialUnetModel
from datasets.nyuv2 import NYUv2DepthDataset

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def make_predictions(model, dataset, device: str, batch_size: int = 5, epsilon: float = 0.0):
    inputs = []
    y_preds = []
    y_trues = []
    aleatoric_vars = []
    epistemic_vars = []
    
    loader = DataLoader(dataset, batch_size=batch_size)

    for data in tqdm(loader):
        images = data['image'].to(device)
        labels = data['label'].cpu()

        images.requires_grad = True
        labels.requires_grad = True

        out = model(images)

        loss = model.loss_fn(out, labels)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = images.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(images, epsilon, data_grad)

        # Predict on the perturbed image
        out = model(perturbed_data)

        out = out.cpu().detach()
        y_true = data['label'].cpu().detach()

        inputs.append(perturbed_data.cpu().detach())

        y_pred = model.loss_fn.mode(out).unsqueeze(dim=1)
        aleatoric_var = model.loss_fn.aleatoric_var(out).unsqueeze(dim=1)
        epistemic_var = model.loss_fn.epistemic_var(out).unsqueeze(dim=1)

        y_preds.append(y_pred)
        y_trues.append(y_true)
        aleatoric_vars.append(aleatoric_var)
        epistemic_vars.append(epistemic_var)

    inputs = torch.cat(inputs, dim=0)
    y_preds = torch.cat(y_preds, dim=0).clip(min=0, max=1)
    y_trues = torch.cat(y_trues, dim=0).clip(min=0, max=1)
    aleatoric_var = torch.cat(aleatoric_vars, dim=0)
    epistemic_var = torch.cat(epistemic_vars, dim=0)
    
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
    
def create_calibration_plot(df: pd.DataFrame, distribution, processes) -> pd.DataFrame:
    
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
    model_checkpoint_path: str,
    datasets: List[Tuple[str, str]],
    result_dir: str,
    device: str,
    processes: int = None,
) -> None:
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=False)

    model = EvidentialUnetModel.load_from_checkpoint(model_checkpoint_path)
    model.to(device)

    for dataset_name, dataset_path in datasets:
        # for noise_level in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]:
        for noise_level in [0.00, 0.02, 0.04]:
            dataset = NYUv2DepthDataset(
                dataset_path=dataset_path,
                normalize=True,
            )

            print(f"Making predictions on {dataset_name}...")
            inputs, y_preds, y_trues, aleatoric_vars, epistemic_vars, combined_vars = make_predictions(
                model=model,
                dataset=dataset,
                batch_size=5,
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
            
            print(f"Creating data for precision-recall plot on {dataset_name}...")
            df_cutoff = create_precision_recall_plot(df)
            df_cutoff.to_csv(result_dir / f"{dataset_name}_{noise_level}_precision_recall.csv", index=False)
            
            print(f"Creating data for calibration plot on {dataset_name}...")
            processes = mp.cpu_count() if processes is None else processes
            df_calibration = create_calibration_plot(df, scipy.stats.norm, processes=processes)
            df_calibration.to_csv(result_dir / f"{dataset_name}_{noise_level}_calibration.csv", index=False)

            print(f"Finished processing dataset `{dataset_name}`!")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint_path", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--monte_carlo_steps", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--processes", type=int, default=None)

    args = parser.parse_args()

    main(
        model_checkpoint_path=args.model_checkpoint_path,
        monte_carlo_steps=args.monte_carlo_steps,
        datasets=[
            ("test", os.path.join(args.dataset_dir, "depth_test.h5")),
            # ("ood", os.path.join(args.dataset_dir, "apolloscape_test.h5")),
        ],
        result_dir=args.result_dir,
        device=args.device,
        processes=args.processes,
    )
