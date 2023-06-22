import torch
import pytorch_lightning as pl
from typing import List
from models.mimo_unet import MimoUnetModel

def repeat_subnetworks(x, num_subnetworks):
    return x.unsqueeze(1).repeat(1, num_subnetworks, 1, 1, 1)

class EnsembleModule(pl.LightningModule):
    def __init__(
        self,
        checkpoint_paths: List[str],
        monte_carlo_steps: int = 0,
    ):
        super().__init__()
        self.models = [MimoUnetModel.load_from_checkpoint(path) for path in checkpoint_paths]
        self.monte_carlo_steps = monte_carlo_steps
        
        for model in self.models:
            model.eval()
            if self.monte_carlo_steps > 0:
                self._activate_mc_dropout(model)

        # todo: check if all models have same loss_fn
        self.loss_fn = self.models[0].loss_fn
            
    @staticmethod
    def _activate_mc_dropout(model: torch.nn.Module):
        """
        Activates MC Dropout for all Dropout layers in the given module.
        Recursively iterates through all submodules.

        Args:
            module: Module to activate MC Dropout for.
        """
        for submodule in model.modules():
            if submodule.__class__.__name__.startswith('Dropout'):
                submodule.train()
                print(f"Activated MC Dropout for {submodule}")
        
    @property
    def num_subnetworks(self):
        return sum(model.num_subnetworks for model in self.models)
        
    def forward(self, x: torch.Tensor):
        """
        Performs a forward pass through all subnetworks of all models.
        Performs Monte Carlo forward passes if self.monte_carlo_steps > 0.
        
        Args:
            x: [B, C_in, H, W]
        Returns:
            p1: [B, S, C_out, H, W]
            p2: [B, S, C_out, H, W]
        """
        p1_list, p2_list = [], []
        
        for model in self.models:
            model.to(self.device)
            x_rep = repeat_subnetworks(x, num_subnetworks=model.num_subnetworks)
            
            for _ in range(max(1, self.monte_carlo_steps)):
                p1, p2 = model(x_rep)
                p1_list.append(p1.cpu())
                p2_list.append(p2.cpu())
        
        p1 = torch.cat(p1_list, dim=1)
        p2 = torch.cat(p2_list, dim=1)
        
        return p1, p2
