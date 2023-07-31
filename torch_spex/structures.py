from collections import defaultdict

import numpy as np
import torch
from typing import Dict, List, Tuple, TypeVar, Callable
import ase
from .neighbor_list import get_neighbor_list

import abc
AtomicStructure = TypeVar('AtomicStructure')

def structure_to_torch(structure : AtomicStructure, device : torch.device = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    :returns:
        Tuple of posititions, species, cell and periodic boundary conditions
    """
    if isinstance(structure, ase.Atoms):
        # dtype is automatically referred from the type in the structure object
        positions = torch.tensor(structure.positions, device=device)
        species = torch.tensor(structure.numbers, device=device)
        cell = torch.tensor(structure.cell.array, device=device)
        pbc = torch.tensor(structure.pbc, device=device)
        return positions, species, cell, pbc
    else:
        raise ValueError("Unknown atom type. We only support ase.Atoms at the moment.")

def build_neighborlist(positions: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor, cutoff : float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    assert positions.device == cell.device
    assert positions.device == pbc.device
    device = positions.device
    # will be replaced with something with GPU support
    pairs_i, pairs_j, cell_shifts = ase.neighborlist.primitive_neighbor_list(
        quantities="ijS",
        positions=positions.detach().cpu().numpy(),
        cell=cell.detach().cpu().numpy(),
        pbc=pbc.detach().cpu().numpy(),
        cutoff=cutoff,
        self_interaction=False,
        use_scaled_positions=False,
    )
    pairs_i = torch.tensor(pairs_i, device=device)
    pairs_j = torch.tensor(pairs_j, device=device)
    cell_shifts = torch.tensor(cell_shifts, device=device)

    pairs = torch.vstack([pairs_i, pairs_j]).T
    centers = torch.arange(len(positions))
    return centers, pairs, cell_shifts


class TransformerBase(metaclass=abc.ABCMeta):
    """
    Abstract class for extracting information of an AtomicStructure objects and processing it
    """
    @abc.abstractmethod
    def __call__(self, structure: AtomicStructure) -> Dict[str, torch.Tensor]:
        pass

class TransformerProperty(TransformerBase):
    """
    Extracts property information out of an AtomicStructure using a function given as input
    """
    def __init__(self, property_name: str, get_property: Callable[[AtomicStructure], Tuple[str, torch.Tensor]]):
        self._get_property = get_property
        self._property_name = property_name

    def __call__(self, structure: AtomicStructure) -> Dict[str, torch.Tensor]:
        return {self._property_name: self._get_property(structure)}

class TransformerNeighborList(TransformerBase):
    """
    Produces a neighbour list and with direction vectors from an AtomicStructure
    """
    def __init__(self, cutoff: float, positions_requires_grad=True, cell_requires_grad=True, device=None):
        self._cutoff = cutoff
        self._positions_requires_grad = positions_requires_grad
        self._cell_requires_grad = cell_requires_grad
        self._structure_index = 0
        self._device = device

    def reset_structure_index(self):
        self._structure_index = 0

    def __call__(self, structure: AtomicStructure) -> Dict[str, torch.Tensor]:
        positions_i, species_i, cell_i, pbc_i = structure_to_torch(structure, device=self._device)
        centers_i, pairs_ij, cell_shifts_ij = build_neighborlist(positions_i, cell_i, pbc_i,  self._cutoff)

        positions_i.requires_grad = self._positions_requires_grad
        cell_i.requires_grad = self._cell_requires_grad

        # cell_shifts_ij needs to be changed to float type to do operations 
        direction_vectors_ij = positions_i[pairs_ij[:, 1]] - positions_i[pairs_ij[:, 0]] + (cell_shifts_ij.to(dtype=cell_i.dtype) @ cell_i)

        structure_index = self._structure_index
        self._structure_index += 1
        return {'structure_centers': torch.tensor([structure_index] * len(centers_i)),
                'structure_pairs': torch.tensor([structure_index] * len(pairs_ij)),
                'positions': positions_i,
                'species': species_i,
                'cell': cell_i.unsqueeze(dim=0),
                'centers': centers_i,
                'pairs': pairs_ij,
                'cell_shifts': cell_shifts_ij,
                'direction_vectors': direction_vectors_ij}



# Temporary Dataset until we have an equistore Dataset
class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self,
                 structures : List[AtomicStructure],
                 transformers : List[TransformerBase]):
        super().__init__()
        self.n_structures = len(structures)
        self._data = defaultdict(list)
        for structure in structures:
            for transformer in transformers:
                data_i = transformer(structure)
                for key in data_i.keys():
                    self._data[key].append(data_i[key])

    def __getitem__(self, idx):
        return {key: self._data[key][idx] for key in self._data.keys()}

    def __len__(self):
        return self.n_structures

def collate_nl(data_list):
    # positions may not be stacked because we need them as leaf nodes to get gradients
    # because concatenating or slicing creates a new node in the autograd graph
    #
    # Autograd grad when concatenating or slicing together the positions for batches
    # positions --DataLoader--> batched_positions
    #    |
    #    |Dataset
    #    |
    #    --> direction_vectors --DataLoader--> batched_direction_vectors
    #
    # Putting the positions into a list does not create a new node

    collated = {key: torch.concatenate([data[key] for data in data_list], dim=0) for key in filter(lambda x : x not in ["positions", "cell"], data_list[0].keys())}
    collated['positions'] = [data["positions"] for data in data_list]
    collated['cell'] = [data["cell"] for data in data_list]
    min_structure_idx = collated['structure_centers'][0].clone()
    collated['structure_centers'] -= min_structure_idx # minimum structure index should be first one
    collated['structure_pairs'] -= min_structure_idx # minimum structure index should be first one
    return collated
