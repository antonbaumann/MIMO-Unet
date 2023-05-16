import torch
import pytorch_lightning as pl
from typing import List
from models.mimo_unet import MimoUnetModel

def repeat_subnetworks(x, num_subnetworks):
    x = x[:, None, :, :, :]
    return x.repeat(1, num_subnetworks, 1, 1, 1)

class EnsembleModule(pl.LightningModule):
    def __init__(
        self,
        checkpoint_paths: List[str],
        monte_carlo_steps: int = 0,
    ):
        super().__init__()
        self.models = [MimoUnetModel.load_from_checkpoint(path) for path in checkpoint_paths]
        self.monte_carlo_steps = monte_carlo_steps
        
        if self.monte_carlo_steps == 0:
            for model in self.models:
                model.eval()
        else:
            self.activate_mc_dropout()
            
    def activate_mc_dropout(self):
        for model in self.models:
            model.eval()
            for m in model.modules():
                # activate training mode for Dropout, Dropout2d, etc. layers
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
        
    @property
    def num_subnetworks(self):
        return sum([model.num_subnetworks] for model in self.models)
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C_in, H, W]
        Returns:
            p1: [B, S, C_out, H, W]
            p2: [B, S, C_out, H, W]
        """
        p1_list, p2_list = [], []

        print(self.models)
        
        for model in self.models:
            x_rep = repeat_subnetworks(x, num_subnetworks=model.num_subnetworks)
            
            if self.monte_carlo_steps == 0:
                p1, p2 = model(x_rep)
                p1_list.append(p1)
                p2_list.append(p2)
            else:
                for _ in range(self.monte_carlo_steps):
                    p1, p2 = model(x_rep)
                    p1_list.append(p1)
                    p2_list.append(p2)
        
        p1 = torch.cat(p1_list, dim=1)
        p2 = torch.cat(p2_list, dim=1)
        
        return p1, p2
