import torch

def softmax_temperature(x: torch.Tensor, temperature=1.0):
    """
    Apply the softmax function with temperature scaling to the input tensor.

    Args:
        tensor: Input tensor to apply softmax to.
        temperature: Temperature scaling factor. Higher values make the output distribution more uniform.
    Returns:
        Softmax output tensor with temperature scaling.
    """
    assert temperature > 0, "Temperature should be positive."

    # Divide the input tensor by the temperature before applying softmax
    return torch.nn.functional.softmax(x / temperature, dim=-1)

class LossBuffer:
    """
    A buffer that stores the losses of the subnetworks and proposes weights to synchoronize learning of the subnetworks
    """
    def __init__(
            self, 
            subnetworks: int,
            temperature: float,
            buffer_size: int,
        ) -> None:
        self.index = 0
        self.temperature = temperature
        self.buffer_size = buffer_size
        self.subnetworks = subnetworks
        self.buffer = torch.zeros(buffer_size, subnetworks)

    def add(self, loss: torch.Tensor) -> None:
        if not self.buffer_size == 0:
            self.buffer[self.index] = loss
            self.index = (self.index + 1) % self.buffer_size
    
    def get_mean(self) -> torch.Tensor:
        if not self.buffer_size == 0:
            return torch.mean(self.buffer, dim=0)
        else:
            return torch.zeros(self.subnetworks)
        
    def get_weights(self) -> torch.Tensor:
        mean = self.get_mean()
        return softmax_temperature(mean, temperature=self.temperature) * len(mean)
