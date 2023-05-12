import json

import pytest

from torch.testing import assert_allclose

import equistore
import numpy as np
import ase.io

from torch_spex.spherical_expansions import VectorExpansion, SphericalExpansion
from torch_spex.structures import Structures

class TestSelection:
    device = "cpu"
    frames = ase.io.read('datasets/rmd17/ethanol1.extxyz', ':1')
    all_species = np.unique(np.hstack([frame.numbers for frame in frames]))
    structures = Structures(frames)
    with open("tests/data/expansion_coeffs-ethanol1_0-hypers.json", "r") as f:
        hypers = json.load(f)

    def test_vector_expansion_coeffs(self):
        tm_ref = equistore.core.io.load_custom_array("tests/data/vector_expansion_coeffs-ethanol1_0-data.npz", equistore.core.io.create_torch_array)
        vector_expansion = VectorExpansion(self.hypers, device="cpu")
        tm = vector_expansion.forward(self.structures)
        equistore.operations.equal(tm_ref, tm)

    def test_spherical_expansion_coeffs(self):
        tm_ref = equistore.core.io.load_custom_array("tests/data/spherical_expansion_coeffs-ethanol1_0-data.npz", equistore.core.io.create_torch_array)
        spherical_expansion_calculator = SphericalExpansion(self.hypers, self.all_species)
        tm  = spherical_expansion_calculator.forward(self.structures)
        equistore.operations.equal(tm_ref, tm)

