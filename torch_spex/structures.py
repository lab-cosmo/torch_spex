import numpy as np
import torch


class Structures:

    # This class essentially takes a list of Atoms objects and converts
    # all the relevant data into torch data structures

    def __init__(self, atoms_list) -> None:
        
        self.n_structures = len(atoms_list)

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

        self.positions = torch.tensor(np.concatenate(positions, axis=0), dtype=torch.get_default_dtype())
        self.cells = cells
        self.structure_indices = np.array(structure_indices)
        self.atomic_species = atomic_species
        self.pbcs = pbcs

    def to(self, device):

        self.positions = self.positions.to(device)

