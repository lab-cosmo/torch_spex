import os

# PR COMMENT: for debug purposes, will be removed before merge
print("os.getcwd()", os.getcwd(), flush=True)
print("os.environ", os.environ, flush=True)

import numpy as np
import torch
from torch_spex.structures import InMemoryDataset, TransformerNeighborList, collate_nl
from torch.utils.data import DataLoader
from torch_spex.spherical_expansions import get_cartesian_vectors
from torch_spex.radial_basis import RadialBasis

import ase.io
import os

# if it is run within an asv environment we prioritize it
# then tox then we assume file is run within torch_spex folder
if os.getenv("ASV_CONF_DIR") is not None:
    TORCH_SPEX_PATH = os.getenv("ASV_CONF_DIR")
elif os.getenv("TOX_WORK_DIR") is not None:
    TORCH_SPEX_PATH = os.path.join(os.getenv("TOX_WORK_DIR"), "../")
else:
    TORCH_SPEX_PATH = "./"

# in the CI we want to be only sure that it works and not run a
# full benchmark
if os.getenv("DRY_RUN"):
    DRY_RUN = True

# When changing the benchmark code the version automatically changes
# to prevent this we manually set the version, for more information
# read https://asv.readthedocs.io/en/stable/benchmarks.html
VERSION = "0.0.0"

class RadialBasisSuiteFloat32:

    version=VERSION

    def setup(self):
        torch.set_default_dtype(torch.float32)
        hypers_radial_basis = {'E_max': 250, 'r_cut': 5.0, 'alchemical': 4}
        frames = ase.io.read(os.path.join(TORCH_SPEX_PATH, 'datasets/alchemical.xyz'), ':1')
        all_species = np.unique(np.hstack([frame.numbers for frame in frames]))
        transformers = [TransformerNeighborList(cutoff=hypers_radial_basis ["r_cut"])]
        dataset = InMemoryDataset(frames, transformers)
        loader = DataLoader(dataset, batch_size=len(frames), collate_fn=collate_nl)
        batch = next(iter(loader))
        batch.pop("positions")
        batch.pop("cell")

        cartesian_vectors = get_cartesian_vectors(**batch)
        #self.batch(["species, cell_shifts, centers, pairs, structure_centers, structure_pairs, direction_vectors)

        bare_cartesian_vectors = cartesian_vectors.values.squeeze(dim=-1)
        self.r = torch.sqrt(
            (bare_cartesian_vectors**2)
            .sum(dim=-1)
        )
        self.samples_metadata = cartesian_vectors.samples  # This can be needed by the radial basis to do alchemical contractions

        self.radial_basis_calculator = RadialBasis(hypers_radial_basis, all_species, device='cpu')
        self.l_max = self.radial_basis_calculator.l_max

    def time_alchemical_dataset(self):
        radial_basis = self.radial_basis_calculator(self.r, self.samples_metadata)

class RadialBasisSuiteFloat64:

    version=VERSION

    def setup(self):
        torch.set_default_dtype(torch.float64)
        hypers_radial_basis = {'E_max': 250, 'r_cut': 5.0, 'alchemical': 4}
        frames = ase.io.read(os.path.join(TORCH_SPEX_PATH, 'datasets/alchemical.xyz'), ':1')
        all_species = np.unique(np.hstack([frame.numbers for frame in frames]))
        transformers = [TransformerNeighborList(cutoff=hypers_radial_basis ["r_cut"])]
        dataset = InMemoryDataset(frames, transformers)
        loader = DataLoader(dataset, batch_size=len(frames), collate_fn=collate_nl)
        batch = next(iter(loader))
        batch.pop("positions")
        batch.pop("cell")

        cartesian_vectors = get_cartesian_vectors(**batch)
        #self.batch(["species, cell_shifts, centers, pairs, structure_centers, structure_pairs, direction_vectors)

        bare_cartesian_vectors = cartesian_vectors.values.squeeze(dim=-1)
        self.r = torch.sqrt(
            (bare_cartesian_vectors**2)
            .sum(dim=-1)
        )
        self.samples_metadata = cartesian_vectors.samples  # This can be needed by the radial basis to do alchemical contractions

        self.radial_basis_calculator = RadialBasis(hypers_radial_basis, all_species, device='cpu')
        self.l_max = self.radial_basis_calculator.l_max


    def time_alchemical_dataset(self):
        radial_basis = self.radial_basis_calculator(self.r, self.samples_metadata)
