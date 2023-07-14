import json

import pytest

import torch

import equistore
from equistore import Labels, TensorBlock, TensorMap
import numpy as np
import ase.io

from torch_spex.spherical_expansions import VectorExpansion, SphericalExpansion
from torch_spex.structures import ase_atoms_to_tensordict

class TestEthanol1SphericalExpansion:
    """
    Tests on the ethanol1 dataset
    """
    device = "cpu"
    frames = ase.io.read('datasets/rmd17/ethanol1.extxyz', ':1')
    all_species = np.unique(np.hstack([frame.numbers for frame in frames]))
    structures = ase_atoms_to_tensordict(frames)
    with open("tests/data/expansion_coeffs-ethanol1_0-hypers.json", "r") as f:
        hypers = json.load(f)

    def test_vector_expansion_coeffs(self):
        tm_ref = equistore.core.io.load_custom_array("tests/data/vector_expansion_coeffs-ethanol1_0-data.npz", equistore.core.io.create_torch_array)
        # we need to sort both computed and reference pair expansion coeffs,
        # because ase.neighborlist can get different neighborlist order for some reasons
        tm_ref = sort_tm(tm_ref)
        vector_expansion = VectorExpansion(self.hypers, self.all_species, device="cpu")
        with torch.no_grad():
            tm = sort_tm(vector_expansion.forward(self.structures))
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

### these util functions will be removed once lab-cosmo/equistore/pull/281 is merged
def native_list_argsort(native_list):
    return sorted(range(len(native_list)), key=native_list.__getitem__)

def sort_tm(tm):
    blocks = []
    for _, block in tm.items():
        values = block.values

        samples_values = block.samples.values
        sorted_idx = native_list_argsort([tuple(row.tolist()) for row in block.samples.values])
        samples_values = samples_values[sorted_idx]
        values = values[sorted_idx]

        components_values = []
        for i, component in enumerate(block.components):
            component_values = component.values
            sorted_idx = native_list_argsort([tuple(row.tolist()) for row in component.values])
            components_values.append( component_values[sorted_idx] )
            values = np.take(values, sorted_idx, axis=i+1)

        properties_values = block.properties.values
        sorted_idx = native_list_argsort([tuple(row.tolist()) for row in block.properties.values])
        properties_values = properties_values[sorted_idx]
        values = values[..., sorted_idx]

        blocks.append(
            TensorBlock(
                values=values,
                samples=Labels(values=samples_values, names=block.samples.names),
                components=[Labels(values=components_values[i], names=component.names) for i, component in enumerate(block.components)],
                properties=Labels(values=properties_values, names=block.properties.names)
            )
        )
    return TensorMap(keys=tm.keys, blocks=blocks)
