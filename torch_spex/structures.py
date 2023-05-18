import numpy as np
import torch
from typing import List
import ase


# Structures = Dict[torch.Tensor]

def Structures(atoms_list : List[ase.Atoms]):
    structure = {}
    structure["n_structures"] = torch.tensor(len(atoms_list))

    positions = []
    cells = []
    structure_indices = []
    atomic_species = []
    pbcs = []

    for structure_index, atoms in enumerate(atoms_list):
        positions.append(atoms.positions)
        cells.append(atoms.cell)
        for _ in range(atoms.positions.shape[0]):
            structure_indices.append(structure_index)
        atomic_species.append(atoms.get_atomic_numbers())
        pbcs.append(atoms.pbc)

    structure["positions"] = torch.tensor(np.concatenate(positions, axis=0), dtype=torch.get_default_dtype())
    structure["cells"] = torch.tensor(cells)
    structure["structure_indices"] = torch.tensor(structure_indices)
    structure["atomic_species"] = torch.tensor(atomic_species)
    structure["pbcs"] = torch.tensor(pbcs)
    return structure

# need to think about this
#    def to(self, device):
#
#        self.positions = self.positions.to(device)

