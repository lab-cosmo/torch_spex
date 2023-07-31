import torch
from typing import List

class AtomicComposition:

    def __init__(self, all_species):
        self.all_species = all_species

    def compute(
        self,
        positions: List[torch.Tensor],
        cells: List[torch.Tensor],
        species: torch.Tensor,
        cell_shifts: torch.Tensor,
        centers: torch.Tensor,
        pairs: torch.Tensor,
        structure_centers: torch.Tensor,
        structure_pairs: torch.Tensor,
        structure_offsets: torch.Tensor
    ):
        n_structures = len(positions)
        composition_features = torch.zeros((n_structures, len(self.all_species)), dtype=torch.get_default_dtype(), device=positions[0].device)
        for i_structure in range(n_structures):
            if i_structure == n_structures-1:
                species_structure = species[structure_offsets[i_structure]:] 
            else:
                species_structure = species[structure_offsets[i_structure]:structure_offsets[i_structure+1]]
            for i_species, atomic_number in enumerate(self.all_species):
                composition_features[i_structure, i_species] = len(torch.where(species_structure == atomic_number)[0])
        return composition_features    
