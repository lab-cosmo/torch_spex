import json

import pytest

import torch

import equistore
import numpy as np
import ase.io

from torch_spex.spherical_expansions import VectorExpansion, SphericalExpansion
from torch_spex.structures import ase_atoms_to_tensordict

class TestEthanol1SphericalExpansion:
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
        # default types are float32 so we set accuracy to 1e-7
        equistore.operations.allclose(tm_ref, tm, atol=1e-7, rtol=1e-7)

    def test_spherical_expansion_coeffs(self):
        tm_ref = equistore.core.io.load_custom_array("tests/data/spherical_expansion_coeffs-ethanol1_0-data.npz", equistore.core.io.create_torch_array)
        spherical_expansion_calculator = SphericalExpansion(self.hypers, self.all_species)
        with torch.no_grad():
            tm = spherical_expansion_calculator.forward(self.structures)
        # default types are float32 so we set accuracy to 1e-7
        equistore.operations.allclose(tm_ref, tm, atol=1e-7, rtol=1e-7)

    def test_spherical_expansion_coeffs_alchemical(self):
        with open("tests/data/expansion_coeffs-ethanol1_0-alchemical-hypers.json", "r") as f:
            hypers = json.load(f)
        tm_ref = equistore.core.io.load_custom_array("tests/data/spherical_expansion_coeffs-ethanol1_0-alchemical-seed0-data.npz", equistore.core.io.create_torch_array)
        torch.manual_seed(0)
        spherical_expansion_calculator = SphericalExpansion(hypers, self.all_species)
        with torch.no_grad():
            tm = spherical_expansion_calculator.forward(self.structures)
        # default types are float32 so we set accuracy to 1e-7
        equistore.operations.allclose(tm_ref, tm, atol=1e-7, rtol=1e-7)

class TestAlchemicalSphericalExpansion:
    device = "cpu"
    frames = ase.io.read('datasets/alchemical.xyz', ':1')
    all_species = np.unique(np.hstack([frame.numbers for frame in frames]))
    structures = ase_atoms_to_tensordict(frames)
    with open("tests/data/expansion_coeffs-alchemical_01-hypers.json", "r") as f:
        hypers = json.load(f)

    def test_vector_expansion_coeffs(self):
        tm_ref = equistore.core.io.load_custom_array("tests/data/vector_expansion_coeffs-alchemical_01-data.npz", equistore.core.io.create_torch_array)
        vector_expansion = VectorExpansion(self.hypers, self.all_species, device="cpu")
        with torch.no_grad():
            tm = vector_expansion.forward(self.structures)
        # default types are float32 so we set accuracy to 1e-7
        equistore.operations.allclose(tm_ref, tm, atol=1e-7, rtol=1e-7)

    def test_spherical_expansion_coeffs(self):
        tm_ref = equistore.core.io.load_custom_array("tests/data/spherical_expansion_coeffs-alchemical_01-data.npz", equistore.core.io.create_torch_array)
        spherical_expansion_calculator = SphericalExpansion(self.hypers, self.all_species)
        with torch.no_grad():
            tm = spherical_expansion_calculator.forward(self.structures)
        # default types are float32 so we set accuracy to 1e-7
        equistore.operations.allclose(tm_ref, tm, atol=1e-7, rtol=1e-7)

    def test_spherical_expansion_coeffs_alchemical(self):
        with open("tests/data/expansion_coeffs-alchemical_01-alchemical-hypers.json", "r") as f:
            hypers = json.load(f)
        tm_ref = equistore.core.io.load_custom_array("tests/data/spherical_expansion_coeffs-alchemical_01-alchemical-seed0-data.npz", equistore.core.io.create_torch_array)
        torch.manual_seed(0)
        spherical_expansion_calculator = SphericalExpansion(hypers, self.all_species)
        with torch.no_grad():
            tm = spherical_expansion_calculator.forward(self.structures)
        # default types are float32 so we set accuracy to 1e-7
        equistore.operations.allclose(tm_ref, tm, atol=1e-7, rtol=1e-7)
