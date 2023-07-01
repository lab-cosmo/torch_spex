import numpy as np
import torch
from typing import Dict, List
import ase

def ase_atoms_to_tensordict(atoms_list : List[ase.Atoms], device : torch.device = "cpu") -> Dict[str, torch.Tensor]:
    """
    dictionary contains
    - **n_structures**: ...,
    - **positions**: ...,
    - **cells**: ...,
    - **structure_indices**: ...,
    - **atomic_species**: ...
    - **cells**: ...,
    - **structure_indices**: ...,
    - **atomic_species**: ...,
    - **pbc**: ...
    """

    atomic_structures = {}

    n_total_atoms = sum([len(atoms) for atoms in atoms_list])
    n_structures = len(atoms_list)
    structures_offsets = np.cumsum([0] + [len(atoms) for atoms in atoms_list])
    atomic_structures["positions"] = torch.empty((n_total_atoms, 3), dtype=torch.get_default_dtype(), device=device)
    atomic_structures["structure_indices"] = torch.empty((n_total_atoms,), dtype=torch.int32)
    atomic_structures["atomic_species"] = torch.empty((n_total_atoms,), dtype=torch.int32)
    atomic_structures["cells"] = torch.empty((n_structures, 3, 3), dtype=torch.get_default_dtype())
    atomic_structures["pbcs"] = torch.empty((n_structures,3), dtype=torch.bool)

    for structure_index, atoms in enumerate(atoms_list):
        atoms_slice = slice(structures_offsets[structure_index], structures_offsets[structure_index+1])
        atomic_structures["positions"][atoms_slice] = torch.tensor(atoms.positions, dtype=torch.get_default_dtype(), device=device)
        atomic_structures["structure_indices"][atoms_slice] = structure_index
        atomic_structures["atomic_species"][atoms_slice] = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32)
        atomic_structures["cells"][structure_index] = torch.tensor(atoms.cell.array, dtype=torch.get_default_dtype())
        atomic_structures["pbcs"][structure_index] = torch.tensor(atoms.pbc, dtype=torch.bool)

    atomic_structures["n_structures"] = torch.tensor(n_structures, dtype=torch.int32)
    return atomic_structures
