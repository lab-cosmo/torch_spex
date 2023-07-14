import json

import pytest

import torch

import equistore
import numpy as np
import ase.io

from torch_spex.spherical_expansions import VectorExpansion, SphericalExpansion
from torch_spex.structures import ase_atoms_to_tensordict

class TestSphericalExpansion:
    device = "cpu"
    frames = ase.io.read('datasets/rmd17/ethanol1.extxyz', ':1')
    all_species = np.unique(np.hstack([frame.numbers for frame in frames]))
    structures = ase_atoms_to_tensordict(frames)
    with open("tests/data/expansion_coeffs-ethanol1_0-hypers.json", "r") as f:
        hypers = json.load(f)

    def test_vector_expansion_coeffs(self):
        tm_ref = equistore.core.io.load_custom_array("tests/data/vector_expansion_coeffs-ethanol1_0-data.npz", equistore.core.io.create_torch_array)
        vector_expansion = VectorExpansion(self.hypers, self.all_species, device="cpu")
        with torch.no_grad():
            tm = vector_expansion.forward(self.structures)
        # Default types are float32 so we cannot get higher accuracy than 1e-7.
        # Because the reference value have been cacluated using float32 and
        # now we using float64 computation the accuracy had to be decreased again
        assert equistore.operations.allclose(tm_ref, tm, atol=1e-5, rtol=1e-5)

    def test_spherical_expansion_coeffs(self):
        tm_ref = equistore.core.io.load_custom_array("tests/data/spherical_expansion_coeffs-ethanol1_0-data.npz", equistore.core.io.create_torch_array)
        spherical_expansion_calculator = SphericalExpansion(self.hypers, self.all_species)
        with torch.no_grad():
            tm = spherical_expansion_calculator.forward(self.structures)
        # Default types are float32 so we cannot get higher accuracy than 1e-7.
        # Because the reference value have been cacluated using float32 and
        # now we using float64 computation the accuracy had to be decreased again
        assert equistore.operations.allclose(tm_ref, tm, atol=1e-5, rtol=1e-5)

    def test_spherical_expansion_coeffs_alchemical(self):
        with open("tests/data/expansion_coeffs-ethanol1_0-alchemical-hypers.json", "r") as f:
            hypers = json.load(f)
        tm_ref = equistore.core.io.load_custom_array("tests/data/spherical_expansion_coeffs-ethanol1_0-alchemical-seed0-data.npz", equistore.core.io.create_torch_array)
        torch.manual_seed(0)
        spherical_expansion_calculator = SphericalExpansion(hypers, self.all_species)
        # Because setting seed seems not be enough to get the same initial combination matrix
        # as in the reference values, we set the combination matrix manually
        with torch.no_grad():
            # wtf? suggested way by torch developers
            # https://discuss.pytorch.org/t/initialize-nn-linear-with-specific-weights/29005/4
            spherical_expansion_calculator.vector_expansion_calculator.radial_basis_calculator.combination_matrix.weight.copy_(torch.tensor(
                    [[-0.00432252,  0.30971584, -0.47518533],
                     [-0.4248946 , -0.22236897,  0.15482073]], dtype=torch.float32))

        with torch.no_grad():
            tm = spherical_expansion_calculator.forward(self.structures)
        # Default types are float32 so we cannot get higher accuracy than 1e-7.
        # Because the reference value have been cacluated using float32 and
        # now we using float64 computation the accuracy had to be decreased again
        assert equistore.operations.allclose(tm_ref, tm, atol=1e-5, rtol=1e-5)
