
import torch
import numpy as np
import copy
import functools
from torch_spex.spherical_expansions import SphericalExpansion
from torch_spex.forces import compute_forces
torch.set_default_dtype(torch.float64)

from torch_spex.structures import InMemoryDataset, TransformerNeighborList, collate_nl
from torch.utils.data import DataLoader
import ase
from ase import io

import equistore
def test_autograd():
    torch.manual_seed(0)

    frames = ase.io.read('tests/datasets/artificial.extxyz', ':')

    hypers = {
        "alchemical": 4,
        "cutoff radius": 4.0,
        "radial basis": {
            "E_max": 300
        }
    }
    all_species = np.unique(np.hstack([frame.numbers for frame in frames]))

    transformers = [TransformerNeighborList(cutoff=hypers["cutoff radius"], positions_requires_grad=True, cell_requires_grad=False)]
    dataset = InMemoryDataset(frames, transformers)
    #collate_fn = functools.partial(collate_nl, position_requires_grad=True, cell_requires_grad=False)
    loader = DataLoader(dataset, batch_size=len(frames), collate_fn=collate_nl)
    input_kwargs = next(iter(loader))

    class Model(torch.nn.Module):

        def __init__(self, hypers, all_species) -> None:
            super().__init__()
            self.all_species = all_species
            self.spherical_expansion_calculator = SphericalExpansion(hypers, all_species)

        def forward(self, spherical_expansion_kwargs, is_compute_forces=True):
            positions = spherical_expansion_kwargs.pop("positions")
            _ = spherical_expansion_kwargs.pop("cell")
            if is_compute_forces:
                spherical_expansion = self.spherical_expansion_calculator(**spherical_expansion_kwargs)
                tm = equistore.sum_over_samples(spherical_expansion, sample_names="center").components_to_properties(["m"]).keys_to_properties(["a_i", "lam", "sigma"])
                energies = torch.sum(tm.block().values, axis=1)

                gradient = torch.autograd.grad(
                    outputs=energies,
                    inputs=positions,
                    grad_outputs=torch.ones_like(energies),
                    retain_graph=False,
                    create_graph=False,
                )
                forces = -torch.concatenate(gradient, dim=0)
                return energies, forces
            else:
                with torch.no_grad():
                    spherical_expansion = self.spherical_expansion_calculator(**spherical_expansion_kwargs)
                    energies = torch.sum(torch.stack([torch.sum(tblock.values) for _, tblock in spherical_expansion.items()]))
                    return energies
    model = Model(hypers, all_species)
    _, backward_forces = model(input_kwargs)

    delta = 1e-6
    structure_forces = []
    for frame in frames:
        for atom_index in range(len(frame)):
            atom_forces = []
            for position_index in range(3):
                frame_minus = copy.deepcopy(frame)
                frame_minus.positions[atom_index, position_index] -= delta
                dataset = InMemoryDataset([frame_minus], transformers)
                loader = DataLoader(dataset, batch_size=1, collate_fn=collate_nl)
                input_kwargs_minus = next(iter(loader))

                frame_plus = copy.deepcopy(frame)
                frame_plus.positions[atom_index, position_index] += delta
                dataset = InMemoryDataset([frame_plus], transformers)
                loader = DataLoader(dataset, batch_size=1, collate_fn=collate_nl)
                input_kwargs_plus = next(iter(loader))

                energy_plus = model(input_kwargs_plus, is_compute_forces=False)
                energy_minus = model(input_kwargs_minus, is_compute_forces=False)
                force = -(energy_plus - energy_minus)/(2*delta)
                atom_forces.append(force)
            atom_forces = torch.tensor(atom_forces)
            structure_forces.append(atom_forces)
    finite_difference_forces = torch.stack(structure_forces, dim=0)

    assert torch.allclose(backward_forces, finite_difference_forces)
