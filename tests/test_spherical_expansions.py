import json

import torch

import metatensor.torch
import ase.io

from torch_spex.spherical_expansions import VectorExpansion, SphericalExpansion
from torch_spex.structures import InMemoryDataset, TransformerNeighborList, collate_nl
from torch.utils.data import DataLoader

class TestEthanol1SphericalExpansion:
    """
    Tests on the ethanol1 dataset
    """
    device = "cpu"
    dtype = torch.float32
    frames = ase.io.read('datasets/rmd17/ethanol1.extxyz', ':1')
    all_species = torch.unique(torch.concatenate([torch.tensor(frame.numbers)
                    for frame in frames]))
    all_species = [int(species) for species in all_species]
    with open("tests/data/expansion_coeffs-ethanol1_0-hypers.json", "r") as f:
        hypers = json.load(f)

    transformers = [TransformerNeighborList(cutoff=hypers["cutoff radius"],
        dtype=dtype, device=device)]

    dataset = InMemoryDataset(frames, transformers)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_nl)
    batch = next(iter(loader))

    def test_vector_expansion_coeffs(self):
        tm_ref = metatensor.torch.load("tests/data/vector_expansion_coeffs-ethanol1_0-data.npz")
        tm_ref = tm_ref.to(device=self.device, dtype=self.dtype)
        # we need to sort both computed and reference pair expansion coeffs,
        # because ase.neighborlist can get different neighborlist order for some reasons
        tm_ref = metatensor.torch.sort(tm_ref)
        vector_expansion = VectorExpansion(self.hypers, self.all_species).to(self.device, self.dtype)
        with torch.no_grad():
            tm = metatensor.torch.sort(vector_expansion.forward(**self.batch))
        # Default types are float32 so we cannot get higher accuracy than 1e-7.
        # Because the reference value have been cacluated using float32 and
        # now we using float64 computation the accuracy had to be decreased again
        assert metatensor.torch.allclose(tm_ref, tm, atol=1e-5, rtol=1e-5)

        vector_expansion_script = torch.jit.script(vector_expansion)
        with torch.no_grad():
            tm_script = metatensor.torch.sort(vector_expansion_script.forward(**self.batch))
        assert metatensor.torch.allclose(tm, tm_script, atol=1e-5,
                rtol=torch.finfo(self.dtype).eps*10)

    def test_spherical_expansion_coeffs(self):
        tm_ref = metatensor.torch.load("tests/data/spherical_expansion_coeffs-ethanol1_0-data.npz")
        tm_ref = tm_ref.to(device=self.device, dtype=self.dtype)
        spherical_expansion_calculator = SphericalExpansion(self.hypers, self.all_species).to(self.device, self.dtype)
        with torch.no_grad():
            tm = spherical_expansion_calculator.forward(**self.batch)
        # Default types are float32 so we cannot get higher accuracy than 1e-7.
        # Because the reference value have been cacluated using float32 and
        # now we using float64 computation the accuracy had to be decreased again
        assert metatensor.torch.allclose(tm_ref, tm, atol=1e-5, rtol=1e-5)

        spherical_expansion_script = torch.jit.script(spherical_expansion_calculator)
        with torch.no_grad():
            tm_script = metatensor.torch.sort(spherical_expansion_script.forward(**self.batch))
        assert metatensor.torch.allclose(tm, tm_script, atol=1e-5,
                rtol=torch.finfo(self.dtype).eps*10)

    def test_spherical_expansion_coeffs_alchemical(self):
        with open("tests/data/expansion_coeffs-ethanol1_0-alchemical-hypers.json", "r") as f:
            hypers = json.load(f)
        tm_ref = metatensor.torch.load("tests/data/spherical_expansion_coeffs-ethanol1_0-alchemical-seed0-data.npz")
        tm_ref = tm_ref.to(device=self.device, dtype=self.dtype)
        torch.manual_seed(0)
        spherical_expansion_calculator = SphericalExpansion(hypers, self.all_species).to(self.device, self.dtype)
        # Because setting seed seems not be enough to get the same initial combination matrix
        # as in the reference values, we set the combination matrix manually
        with torch.no_grad():
            # wtf? suggested way by torch developers
            # https://discuss.pytorch.org/t/initialize-nn-linear-with-specific-weights/29005/4
            spherical_expansion_calculator.vector_expansion_calculator.radial_basis_calculator.combination_matrix.weight.copy_(torch.tensor(
                    [[-0.00432252,  0.30971584, -0.47518533],
                     [-0.4248946 , -0.22236897,  0.15482073]],
                    device=self.device, dtype=self.dtype))

        with torch.no_grad():
            tm = spherical_expansion_calculator.forward(**self.batch)
        # Default types are float32 so we cannot get higher accuracy than 1e-7.
        # Because the reference value have been cacluated using float32 and
        # now we using float64 computation the accuracy had to be decreased again
        assert metatensor.torch.allclose(tm_ref, tm, atol=1e-5, rtol=1e-5)

class TestArtificialSphericalExpansion:
    """
    Tests on the artificial dataset
    """
    device = "cpu"
    dtype = torch.float32
    frames = ase.io.read('tests/datasets/artificial.extxyz', ':')
    all_species = torch.unique(torch.concatenate([torch.tensor(frame.numbers)
                    for frame in frames]))
    all_species = [int(species) for species in all_species]
    with open("tests/data/expansion_coeffs-artificial-hypers.json", "r") as f:
        hypers = json.load(f)

    transformers = [TransformerNeighborList(cutoff=hypers["cutoff radius"],
        dtype=dtype, device=device)]
    dataset = InMemoryDataset(frames, transformers)
    loader = DataLoader(dataset, batch_size=len(frames), collate_fn=collate_nl)
    batch = next(iter(loader))

    def test_vector_expansion_coeffs(self):
        tm_ref = metatensor.torch.load("tests/data/vector_expansion_coeffs-artificial-data.npz")
        tm_ref = tm_ref.to(device=self.device, dtype=self.dtype)
        tm_ref = metatensor.torch.sort(tm_ref)
        vector_expansion = VectorExpansion(self.hypers, self.all_species).to(self.device, self.dtype)
        with torch.no_grad():
            tm = metatensor.torch.sort(vector_expansion.forward(**self.batch))
        assert metatensor.torch.allclose(tm_ref, tm, atol=1e-5, rtol=1e-5)

    def test_spherical_expansion_coeffs(self):
        tm_ref = metatensor.torch.load("tests/data/spherical_expansion_coeffs-artificial-data.npz")
        tm_ref = tm_ref.to(device=self.device, dtype=self.dtype)
        spherical_expansion_calculator = SphericalExpansion(self.hypers, self.all_species).to(self.device, self.dtype)
        with torch.no_grad():
            tm = spherical_expansion_calculator.forward(**self.batch)
        # The absolute accuracy is a bit smaller than in the ethanol case
        # I presume it is because we use 5 frames instead of just one
        assert metatensor.torch.allclose(tm_ref, tm, atol=3e-5, rtol=1e-5)

    def test_spherical_expansion_coeffs_artificial(self):
        with open("tests/data/expansion_coeffs-artificial-alchemical-hypers.json", "r") as f:
            hypers = json.load(f)
        tm_ref = metatensor.torch.load("tests/data/spherical_expansion_coeffs-artificial-alchemical-seed0-data.npz")
        tm_ref = tm_ref.to(device=self.device, dtype=self.dtype)
        spherical_expansion_calculator = SphericalExpansion(hypers, self.all_species).to(self.device, self.dtype)
        with torch.no_grad():
            spherical_expansion_calculator.vector_expansion_calculator.radial_basis_calculator.combination_matrix.weight.copy_(
                torch.tensor(
                    [[-0.00432252,  0.30971584, -0.47518533],
                     [-0.4248946 , -0.22236897,  0.15482073]],
                    device=self.device, dtype=self.dtype
                )
            )
        with torch.no_grad():
            tm = spherical_expansion_calculator.forward(**self.batch)
        assert metatensor.torch.allclose(tm_ref, tm, atol=1e-5, rtol=1e-5)
