import torch
from typing import Optional


def apply_input_transform(
        image: torch.Tensor,
        label: torch.Tensor,
        mask: Optional[torch.Tensor],
        num_subnetworks: int,
        input_repetition_probability: float = 0.0,
        batch_repetitions: int = 1,
    ):
    """
    Apply input transformations to the input data.
    - batch repetition
    - input repetition

    Args:
        image: [B, C_image, H, W]
        label: [B, C_label, H, W]
        mask: [B, 1, H, W]
    Returns:
        [B, S, C_image, H, W], [B, S, C_label, H, W], [B, S, 1, H, W]
    """
    B, _, _, _ = image.shape

    main_shuffle = torch.randperm(B, device=image.device).repeat(batch_repetitions)
    to_shuffle = int(main_shuffle.shape[0] * (1. - input_repetition_probability))

    shuffle_indices = [
        torch.cat(
            (main_shuffle[:to_shuffle][torch.randperm(to_shuffle)], main_shuffle[to_shuffle:]), 
            dim=0,
        ) for _ in range(num_subnetworks)
    ]

    image_transformed = torch.stack(
        [torch.index_select(image, 0, indices) for indices in shuffle_indices], 
        dim=1,
    )
    label_transformed = torch.stack(
        [torch.index_select(label, 0, indices) for indices in shuffle_indices],
        dim=1,
    )
    mask_transformed = torch.stack(
        [torch.index_select(mask, 0, indices) for indices in shuffle_indices],
        dim=1,
    ) if mask is not None else None
    return image_transformed, label_transformed, mask_transformed

def repeat_subnetworks(x: torch.Tensor, num_subnetworks: int):
        """
        Repeat the input tensor along the subnetwork dimension.

        Args:
            x: [B, C, H, W]
        Returns:
            x: [B, S, C, H, W]
        """
        x = x[:, None, :, :, :]
        return x.repeat(1, num_subnetworks, 1, 1, 1)

def flatten_subnetwork_dimension(x: torch.Tensor):
        """
        Collapse the subnetwork dimension into the batch dimension.

        Args:
            x: [B // S, S, C, H, W]
        Returns:
            x: [B, C, H, W]
        """
        B_, S, C, H, W = x.shape
        # [B // S, S, C, H, W] -> [B, C, H, W]
        return x.view(B_ * S, C, H, W)

def compute_uncertainties(criterion, y_preds, log_params):
    """
    Compute uncertainties from the predicted mean and log parameters.

    Args:
        criterion: Loss function.
        y_preds: [B, S, C, H, W]
        log_params: [B, S, C, H, W]
    Returns:
        mean: [B, C, H, W]
        aleatoric_variance: [B, C, H, W]
        epistemic_variance: [B, C, H, W]
    """
    _, S, _, _, _ = y_preds.shape
    
    mean = criterion.mode(y_preds, log_params).mean(dim=1)
    stds = criterion.std(y_preds, log_params)
    aleatoric_variance = torch.square(stds).mean(dim=1)
    
    if S > 1:
        y_preds_mean = y_preds.mean(dim=1, keepdims=True)
        epistemic_variance = torch.square(y_preds - y_preds_mean).sum(dim=1) / (S - 1)
    else:
        epistemic_variance = torch.zeros_like(aleatoric_variance)
        
    return mean, aleatoric_variance, epistemic_variance