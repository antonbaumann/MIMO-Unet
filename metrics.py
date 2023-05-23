from typing import List, Dict, Optional
import torch
import torchmetrics


def get_metric(metric: str):
    if metric == "mae":
        return torchmetrics.functional.mean_absolute_error
    elif metric == "mse":
        return torchmetrics.functional.mean_squared_error
    elif metric == "r2":
        return torchmetrics.functional.r2_score
    elif metric == "mape":
        return torchmetrics.functional.mean_absolute_percentage_error
    else:
        raise ValueError(f"Unknown metric: {metric}")
    

def compute_regression_metrics(
    y_hat: torch.Tensor, 
    y: torch.Tensor, 
    metrics: Optional[List[str]] = ['r2', 'mae', 'mse'],
) -> Dict[str, float]:
    y = y.detach()
    y_hat = y_hat.detach()

    metric_dict = dict()
    for metric in metrics:
        metric_dict[metric] = get_metric(metric)(y_hat, y)

    return metric_dict  
