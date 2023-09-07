import json

import torch

import metatensor.torch
from metatensor.torch import Labels, TensorBlock, TensorMap
import numpy as np
import ase.io

from torch_spex.spherical_expansions import VectorExpansion, SphericalExpansion
from torch_spex.structures import InMemoryDataset, TransformerNeighborList, collate_nl
from torch.utils.data import DataLoader


class TestEthanol1SphericalExpansion:
    """
    Tests on the ethanol1 dataset
    """
    device = "cpu"
    frames = ase.io.read('datasets/rmd17/ethanol1.extxyz', ':1')
    all_species = list(np.unique([frame.numbers for frame in frames]))
    all_species = [int(species) for species in all_species]
    with open("tests/data/expansion_coeffs-ethanol1_0-hypers.json", "r") as f:
        hypers = json.load(f)

    transformers = [TransformerNeighborList(cutoff=hypers["cutoff radius"])]
    dataset = InMemoryDataset(frames, transformers)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_nl)
    batch = next(iter(loader))

    def test_vector_expansion_coeffs(self):
        vector_expansion = torch.jit.script(VectorExpansion(self.hypers, self.all_species, device=self.device))
        vector_expansion.forward(**self.batch)

    def test_spherical_expansion_coeffs(self):
        spherical_expansion_calculator = torch.jit.script(SphericalExpansion(self.hypers, self.all_species, device=self.device))
        spherical_expansion_calculator.forward(**self.batch)

    def test_spherical_expansion_coeffs_alchemical(self):
        with open("tests/data/expansion_coeffs-ethanol1_0-alchemical-hypers.json", "r") as f:
            hypers = json.load(f)
        spherical_expansion_calculator = torch.jit.script(SphericalExpansion(hypers, self.all_species, device=self.device))
        spherical_expansion_calculator.forward(**self.batch)

class TestArtificialSphericalExpansion:
    """
    Tests on the artificial dataset
    """
    device = "cpu"
    frames = ase.io.read('tests/datasets/artificial.extxyz', ':')
    all_species = list(np.unique(np.hstack([frame.numbers for frame in frames])))
    all_species = [int(species) for species in all_species]
    with open("tests/data/expansion_coeffs-artificial-hypers.json", "r") as f:
        hypers = json.load(f)

    transformers = [TransformerNeighborList(cutoff=hypers["cutoff radius"])]
    dataset = InMemoryDataset(frames, transformers)
    loader = DataLoader(dataset, batch_size=len(frames), collate_fn=collate_nl)
    batch = next(iter(loader))

    def test_vector_expansion_coeffs(self):
        vector_expansion = torch.jit.script(VectorExpansion(self.hypers, self.all_species, device=self.device))
        vector_expansion.forward(**self.batch)

    def test_spherical_expansion_coeffs(self):
        spherical_expansion_calculator = torch.jit.script(SphericalExpansion(self.hypers, self.all_species, device=self.device))
        spherical_expansion_calculator.forward(**self.batch)

    def test_spherical_expansion_coeffs_artificial(self):
        with open("tests/data/expansion_coeffs-artificial-alchemical-hypers.json", "r") as f:
            hypers = json.load(f)
        spherical_expansion_calculator = torch.jit.script(SphericalExpansion(hypers, self.all_species, device=self.device))
        spherical_expansion_calculator.forward(**self.batch)
