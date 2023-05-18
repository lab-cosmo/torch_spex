import json
import pytest

import torch

import equistore
import numpy as np
import ase.io

from torch_spex.spherical_expansions import VectorExpansion, SphericalExpansion
from torch_spex.structures import Structures

#from torch_spex.spliner import Structures

class TestTorchScript:
    device = "cpu"
    frames = ase.io.read('datasets/rmd17/ethanol1.extxyz', ':1')
    all_species = np.unique(np.hstack([frame.numbers for frame in frames]))
    structures = Structures(frames)
    with open("tests/data/expansion_coeffs-ethanol1_0-hypers.json", "r") as f:
        hypers = json.load(f)

    def test_vector_expansion_coeffs(self):
        torch.manual_seed(0)
        spherical_expansion_calculator = SphericalExpansion(self.hypers, self.all_species)
        spherical_expansion_calculator.forward(self.structures)
        script_module = torch.jit.trace(spherical_expansion_calculator, (self.structures,))
        torch.jit.script(script_module)
