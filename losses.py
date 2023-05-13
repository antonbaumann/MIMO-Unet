import torch
from abc import ABC, abstractmethod

class UncertaintyLoss(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, y_hat, log_variance, y, mask):
        pass

    @abstractmethod
    def std(self, mu, log_variance):
        pass

    @abstractmethod
    def mode(self, mu, log_variance):
        pass

    @classmethod
    def from_name(cls, name: str) -> "UncertaintyLoss":
        if name == "gaussian_nll":
            return GaussianNLL()
        elif name == "laplace_nll":
            return LaplaceNLL()
        else:
            raise ValueError(f"Unknown loss function: {name}")


class GaussianNLL(UncertaintyLoss):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.min_log_variance = torch.log(torch.tensor(eps))

    def forward(
        self, 
        y_hat: torch.tensor, 
        log_variance: torch.tensor, 
        y: torch.tensor,
        mask: torch.tensor = None,
        reduce_mean: bool = True,
    ):
        """Negative log-likelihood for a Gaussian distribution.
        Adding weight decay yields the negative log posterior.

        Args:
            y_hat: Predicted mean
            log_variance: Predicted log variance
            y: Target
        Returns:
            Negative log-likelihood for a Gaussian distribution
        """

        # Clip log variance to avoid possible division by zero
        log_variance = torch.max(log_variance, self.min_log_variance)

        diff = y_hat - y
        diff[~torch.isfinite(diff)] = 0.0
        
        loss = log_variance + diff ** 2 / torch.exp(log_variance)

        if mask is not None:
            loss = loss * mask

        if reduce_mean:
            return torch.mean(loss)
        return loss


    def std(
        self, 
        mu: torch.tensor, 
        log_variance: torch.tensor
    ):
        return torch.exp(log_variance) ** 0.5

    def mode(
        self, 
        mu: torch.tensor, 
        log_variance: torch.tensor,
    ):
        return mu


class LaplaceNLL(UncertaintyLoss):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.min_log_scale = torch.log(torch.tensor(eps))
        
    def forward(
        self, 
        y_hat: torch.tensor, 
        log_scale: torch.tensor, 
        y: torch.tensor,
        mask: torch.tensor = None,
        reduce_mean: bool = True,
    ):
        """Negative log-likelihood for a Laplace distribution.
        Adding weight decay yields the negative log posterior.

        Args:
            y_hat: Predicted mean
            log_scale: Predicted log scale
            y: Target
        Returns:
            Negative log-likelihood for a Laplace distribution
        """

        # Clip log scale to avoid possible division by zero
        log_scale = torch.max(log_scale, self.min_log_scale)

        diff = y_hat - y
        diff[~torch.isfinite(diff)] = 0.0

        loss = log_scale + diff.abs() / torch.exp(log_scale)

        if mask is not None:
            loss = loss * mask

        if reduce_mean:
            return torch.mean(loss)
        return loss

    def std(self, mu, log_scale):
        return torch.exp(log_scale) * (2 ** 0.5)

    def mode(self, mu, log_scale):
        return mu