import torch
import copy
from torch_spex.spherical_expansions import SphericalExpansion
from torch_spex.forces import compute_forces
torch.set_default_dtype(torch.float64)

from torch_spex.structures import ase_atoms_to_tensordict
import ase
from ase import io


def test_autograd():

    structure = ase.io.read('datasets/rmd17/ethanol1.extxyz', ':1')[0]
    hypers = {
        "alchemical": 4,
        "cutoff radius": 4.0,
        "radial basis": {
            "type": "le",
            "E_max": 300
        }
    }
    all_species = [1, 6, 8]

    class Model(torch.nn.Module):

        def __init__(self, hypers, all_species) -> None:
            super().__init__()
            self.all_species = all_species
            self.spherical_expansion_calculator = SphericalExpansion(hypers, all_species)

        def forward(self, structures):

            # print("Transforming structures")
            structures = ase_atoms_to_tensordict(structures)
            structures["positions"].requires_grad = True
            spherical_expansion = self.spherical_expansion_calculator(structures)
            energies = torch.sum(torch.stack([torch.sum(tblock.values) for _, tblock in spherical_expansion.items()]))
            forces = compute_forces(energies, structures["positions"], is_training=False)

            return energies, forces
    
    model = Model(hypers, all_species)

    delta = 1e-6

    _, backward_forces = model([structure])

    structure_forces = []
    for atom_index in range(len(structure.positions)):
        atom_forces = []
        for position_index in range(3):
            structure_plus = copy.deepcopy(structure)
            structure_minus = copy.deepcopy(structure)
            structure_plus.positions[atom_index, position_index] += delta
            structure_minus.positions[atom_index, position_index] -= delta
            energy_plus, _ = model([structure_plus])
            energy_minus, _ = model([structure_minus])
            force = -(energy_plus - energy_minus)/(2*delta)
            atom_forces.append(force)
        atom_forces = torch.tensor(atom_forces)
        structure_forces.append(atom_forces)
    finite_difference_forces = torch.stack(structure_forces, dim=0)

    assert torch.allclose(backward_forces, finite_difference_forces)
    print("Finite differences check passed successfully!")


if __name__ == "__main__":
    pass


