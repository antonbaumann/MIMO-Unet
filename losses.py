import torch
from abc import ABC, abstractmethod

class UncertaintyLoss(torch.nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def forward(self, y_hat, log_variance, y, mask) -> torch.Tensor:
        pass

    @abstractmethod
    def std(self, mu, log_variance) -> torch.Tensor:
        pass

    @abstractmethod
    def mode(self, mu, log_variance) -> torch.Tensor:
        pass

    @abstractmethod
    def calculate_dist_param(self, std: torch.Tensor, *, log: bool = False) -> torch.Tensor:
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
    def __init__(self, eps_min: float = 1e-5, eps_max: float = 1e3):
        super().__init__()
        self.eps_min = eps_min
        self.eps_max = eps_max

    def forward(
        self, 
        y_hat: torch.tensor, 
        log_variance: torch.tensor, 
        y: torch.tensor,
        *,
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
        diff = y_hat - y

        variance = torch.exp(log_variance).clone()
        with torch.no_grad():
            variance.clamp_(min=self.eps_min, max=self.eps_max)
        
        loss = torch.log(variance) + diff ** 2 / variance

        if mask is not None:
            loss = loss * mask

        if reduce_mean:
            return torch.mean(loss)
        return loss


    def std(
        self, 
        mu: torch.Tensor, 
        log_variance: torch.Tensor
    ):
        return torch.exp(log_variance) ** 0.5

    def mode(
        self, 
        mu: torch.Tensor, 
        log_variance: torch.Tensor,
    ):
        return mu
    
    def calculate_dist_param(
        self, 
        std: torch.Tensor,
        *,
        log: bool = False,
    ):
        """
        Calculate the distribution parameter based on the provided standard deviation.

        Args:
            std: The tensor containing the standard deviation values.
            log: If set to True, return the natural logarithm of the calculated parameter.

        Returns:
            A tensor with the calculated distribution parameter.
        """
        param = std ** 2
        param = param.clone()

        with torch.no_grad():
            param.clamp_(min=self.eps_min, max=self.eps_max)

        if log:
            param = torch.log(param)

        return param


class LaplaceNLL(UncertaintyLoss):
    def __init__(self, eps_min: float = 1e-5, eps_max: float = 1e3):
        super().__init__()
        self.eps_min = eps_min
        self.eps_max = eps_max
        
    def forward(
        self, 
        y_hat: torch.Tensor, 
        log_scale: torch.Tensor, 
        y: torch.Tensor,
        *,
        mask: torch.Tensor = None,
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
        diff = y_hat - y

        scale = torch.exp(log_scale).clone()
        with torch.no_grad():
            scale.clamp_(min=self.eps_min, max=self.eps_max)

        loss = torch.log(scale) + diff.abs() / scale

        if mask is not None:
            loss = loss * mask.unsqueeze(dim=1)

        if reduce_mean:
            return torch.mean(loss)
        return loss

    def std(self, mu, log_scale):
        return torch.exp(log_scale) * (2 ** 0.5)

    def mode(self, mu, log_scale):
        return mu
    
    def calculate_dist_param(self, std: torch.Tensor, *, log: bool = False) -> torch.Tensor:
        """
        Calculate the distribution parameter based on the provided standard deviation.

        Args:
            std: The tensor containing the standard deviation values.
            log: If set to True, return the natural logarithm of the calculated parameter.

        Returns:
            A tensor with the calculated distribution parameter.
        """
        param = std / (2 ** 0.5)
        param = param.clone()

        with torch.no_grad():
            param.clamp_(min=self.eps_min, max=self.eps_max)

        if log:
            param = torch.log(param)

        return param
