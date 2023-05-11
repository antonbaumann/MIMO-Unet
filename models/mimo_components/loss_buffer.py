import torch

class LossBuffer:
    def __init__(
            self, 
            buffer_size: int = 10, 
            subnetworks: int = 1
        ) -> None:
        self.i = 0
        self.buffer_size = buffer_size
        self.buffer = torch.zeros(buffer_size, subnetworks)

    def add(self, loss: torch.Tensor) -> None:
        self.buffer[self.i] = loss
        self.i = (self.i + 1) % self.buffer_size
    
    def get_mean(self) -> torch.Tensor:
        return torch.mean(self.buffer, dim=0)
    
    def get_weights(self) -> torch.Tensor:
        mean = self.get_mean()
        if mean.sum() == 0:
            return torch.ones_like(mean)
        return (1 - mean / torch.sum(mean)) * len(mean)

