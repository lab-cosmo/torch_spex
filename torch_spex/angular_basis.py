import numpy as np
import torch
import sphericart_torch


class AngularBasis(torch.nn.Module):

    def __init__(self, l_max) -> None:
        super().__init__()
        
        self.l_max = l_max
        self.sh_calculator = sphericart_torch.SphericalHarmonics(l_max, normalized=True)

    def forward(self, xyz):

        sh, _ = self.sh_calculator.compute(xyz, gradients=False)
        sh = [sh[:, l**2:(l+1)**2] for l in range(self.l_max+1)]
        return sh
