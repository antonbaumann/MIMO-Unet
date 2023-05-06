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
    elif metric == "ssim":
        return torchmetrics.functional.structural_similarity_index_measure
    elif metric == "multiscalessim":
        return torchmetrics.functional.multiscale_structural_similarity_index_measure
    elif metric == "mape":
        return torchmetrics.functional.mean_absolute_percentage_error
    else:
        raise ValueError(f"Unknown metric: {metric}")
    

def compute_regression_metrics(
    y_hat: torch.Tensor, 
    y: torch.Tensor, 
    metrics: Optional[List[str]] = ['r2', 'mae', 'mse', 'ssim', 'mape'],
) -> Dict[str, float]:
    y_flat = y.detach().flatten()
    y_hat_flat = y_hat.detach().flatten()

    metric_dict = dict()
    for metric in metrics:
        metric_dict[metric] = get_metric(metric)(y_hat_flat, y_flat)

    return metric_dict  


def _log_metrics(self, y_hat, y, mode: str, batch_idx: int) -> None:
        if mode != "val" and (batch_idx + 1) % self.trainer.log_every_n_steps == 0:
            return

        # global metrics
        metric_filter = lambda attr: attr.startswith("metric_") and attr.count("/") == 1 and mode in attr
        metric_name_list = filter(metric_filter, dir(self))

        for metric_name in metric_name_list:
            metric = getattr(self, metric_name)
            metric(y_hat.detach().flatten(), y.detach().flatten())
            self.log(
                metric_name,
                metric,
                on_step=True,
                on_epoch=True,
                metric_attribute=metric_name,
                batch_size=self.trainer.datamodule.batch_size,
            )
