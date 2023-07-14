import copy

import numpy as np
import torch
from equistore import TensorMap, Labels, TensorBlock
import sphericart.torch

from .radial_basis import RadialBasis
from .neighbor_list import get_neighbor_list
from typing import Dict, List

class SphericalExpansion(torch.nn.Module):
    """
    The spherical expansion coefficients summed over all neighbours.

    .. math::

         \sum_j c^{l}_{Aija_ia_j, m, n} = c^{l}_{Aia_ia_j, m, n}
         --reorder--> c^{a_il}_{Ai, m, a_jn}

    where:
    - **A**: index atomic structure,
    - **i**: index of central atom,
    - **j**: index of neighbor atom,
    - **a_i**: species of central atom,
    - **a_j**: species of neighbor atom or pseudo species,
    - **n**: radial channel corresponding to n'th radial basis function,
    - **l**: degree of spherical harmonics,
    - **m**: order of spherical harmonics

    The indices of the coefficients are written to show the storage in an
    equistore.TensorMap object

    .. math::

         c^{keys}_{samples, components, properties}

    :param hypers:
        - **cutoff radius**: cutoff for the neighborlist
        - **radial basis**: smooth basis optimizing Rayleight quotients [lle]_
          - **E_max** energy cutoff for the eigenvalues of the eigenstates
        - **alchemical**: number of pseudo species to reduce the species channels to

    .. [lle]
        Bigi, Filippo, et al. "A smooth basis for atomistic machine learning."
        The Journal of Chemical Physics 157.23 (2022): 234101.
        https://doi.org/10.1063/5.0124363

    >>> import numpy as np
    >>> from ase.build import molecule
    >>> from torch_spex.structures import ase_atoms_to_tensordict
    >>> from torch_spex.spherical_expansions import SphericalExpansion
    >>> hypers = {
    ...     "cutoff radius": 3,
    ...     "radial basis": {
    ...         "E_max": 20
    ...     },
    ...     "alchemical": 1,
    ... }
    >>> h2o = molecule("H2O")
    >>> spherical_expansion = SphericalExpansion(hypers, [1,8], device="cpu")
    >>> atomic_structures = ase_atoms_to_tensordict([h2o])
    >>> spherical_expansion.forward(atomic_structures)
    TensorMap with 2 blocks
    keys: a_i  lam  sigma
           1    0     1
           8    0     1

    """

    def __init__(self, hypers: Dict, all_species: List[int], device: str ="cpu") -> None:
        super().__init__()

        self.hypers = hypers
        self.all_species = np.array(all_species, dtype=np.int32)  # convert potential list to np.array
        self.vector_expansion_calculator = VectorExpansion(hypers, self.all_species, device=device)

        if "alchemical" in self.hypers:
            self.is_alchemical = True
            self.n_pseudo_species = self.hypers["alchemical"]
        else:
            self.is_alchemical = False

    def forward(self, structures: Dict[str, torch.Tensor]):

        expanded_vectors = self.vector_expansion_calculator(structures)
        samples_metadata = expanded_vectors.block(l=0).samples

        s_metadata = torch.LongTensor(samples_metadata["structure"].copy())  # Copy to suppress torch warning about non-writeability
        i_metadata = torch.LongTensor(samples_metadata["center"].copy())

        n_species = len(self.all_species)
        species_to_index = {atomic_number : i_species for i_species, atomic_number in enumerate(self.all_species)}

        unique_s_i_indices = torch.stack((structures["structure_indices"], structures["center_number"]), dim=1)
        s_i_metadata_to_unique = structures["structure_offsets"][s_metadata] + i_metadata

        l_max = self.vector_expansion_calculator.l_max
        n_centers = len(unique_s_i_indices)  # total number of atoms in this batch of structures

        densities = []
        if self.is_alchemical:
            density_indices = torch.LongTensor(s_i_metadata_to_unique)
            for l in range(l_max+1):
                expanded_vectors_l = expanded_vectors.block(l=l).values
                densities_l = torch.zeros(
                    (n_centers, expanded_vectors_l.shape[1], expanded_vectors_l.shape[2]),
                    dtype = expanded_vectors_l.dtype,
                    device = expanded_vectors_l.device
                )
                densities_l.index_add_(dim=0, index=density_indices.to(expanded_vectors_l.device), source=expanded_vectors_l)
                densities_l = densities_l.reshape((n_centers, 2*l+1, -1))
                densities.append(densities_l)
            species = -np.arange(self.n_pseudo_species)
        else:
            aj_metadata = samples_metadata["species_neighbor"]
            aj_shifts = np.array([species_to_index[aj_index] for aj_index in aj_metadata])
            density_indices = torch.LongTensor(s_i_metadata_to_unique*n_species+aj_shifts)

            for l in range(l_max+1):
                expanded_vectors_l = expanded_vectors.block(l=l).values
                densities_l = torch.zeros(
                    (n_centers*n_species, expanded_vectors_l.shape[1], expanded_vectors_l.shape[2]),
                    dtype = expanded_vectors_l.dtype,
                    device = expanded_vectors_l.device
                )
                densities_l.index_add_(dim=0, index=density_indices.to(expanded_vectors_l.device), source=expanded_vectors_l)
                densities_l = densities_l.reshape((n_centers, n_species, 2*l+1, -1)).swapaxes(1, 2).reshape((n_centers, 2*l+1, -1))  # need to swap n, a indices which are in the wrong order
                densities.append(densities_l)
            species = self.all_species

        # constructs the TensorMap object
        ai_new_indices = structures["atomic_species"]
        labels = []
        blocks = []
        for l in range(l_max+1):
            densities_l = densities[l]
            vectors_l_block = expanded_vectors.block(l=l)
            vectors_l_block_components = vectors_l_block.components
            vectors_l_block_n = np.arange(len(np.unique(vectors_l_block.properties["n"])))  # Need to be smarter to optimize
            for a_i in self.all_species:
                where_ai = torch.LongTensor(np.where(ai_new_indices == a_i)[0]).to(densities_l.device)
                densities_ai_l = torch.index_select(densities_l, 0, where_ai)
                if self.normalize:
                    densities_ai_l *= self.normalization_factor
                labels.append([a_i, l, 1])
                blocks.append(
                    TensorBlock(
                        values = densities_ai_l,
                        samples = Labels(
                            names = ["structure", "center"],
                            values = unique_s_i_indices.numpy()[where_ai.cpu().numpy()]
                        ),
                        components = vectors_l_block_components,
                        properties = Labels(
                            names = ["a1", "n1", "l1"],
                            values = np.stack(
                                [
                                    np.repeat(species, vectors_l_block_n.shape[0]),
                                    np.tile(vectors_l_block_n, species.shape[0]),
                                    l*np.ones((densities_ai_l.shape[2],), dtype=np.int32)
                                ],
                                axis=1
                            )
                        )
                    )
                )

        spherical_expansion = TensorMap(
            keys = Labels(
                names = ["a_i", "lam", "sigma"],
                values = np.array(labels, dtype=np.int32)
            ),
            blocks = blocks
        )

        return spherical_expansion


class VectorExpansion(torch.nn.Module):
    """
    The spherical expansion coefficients for each neighbour

    .. math::

        c^{l}_{Aija_ia_j,m,n}

    where:
    - **A**: index atomic structure,
    - **i**: index of central atom,
    - **j**: index of neighbor atom,
    - **a_i**: species of central atom,
    - **a_j**: species of neighbor aotm,
    - **n**: radial channel corresponding to n'th radial basis function,
    - **l**: degree of spherical harmonics,
    - **m**: order of spherical harmonics

    The indices of the coefficients are written to show the storage in an
    equistore.TensorMap object

    .. math::

         c^{keys}_{samples, components, properties}

    """

    def __init__(self, hypers: Dict, all_species, device: str = "cpu") -> None:
        super().__init__()

        self.hypers = hypers
        self.normalize = True if "normalize" in hypers else False
        if self.normalize:
            avg_num_neighbors = hypers["normalize"]
            self.normalization_factor = 1.0/np.sqrt(avg_num_neighbors)
        # radial basis needs to know cutoff so we pass it, as well as whether to normalize or not
        hypers_radial_basis = copy.deepcopy(hypers["radial basis"])
        hypers_radial_basis["r_cut"] = hypers["cutoff radius"]
        hypers_radial_basis["normalize"] = self.normalize
        if "alchemical" in self.hypers:
            self.is_alchemical = True
            self.n_pseudo_species = self.hypers["alchemical"]
            hypers_radial_basis["alchemical"] = self.hypers["alchemical"]
        else:
            self.is_alchemical = False
        self.radial_basis_calculator = RadialBasis(hypers_radial_basis, all_species, device=device)
        self.l_max = self.radial_basis_calculator.l_max
        self.spherical_harmonics_calculator = sphericart.torch.SphericalHarmonics(self.l_max, normalized=True)
        self.spherical_harmonics_split_list = [(2*l+1) for l in range(self.l_max+1)]

    def forward(self, structures: Dict[str, torch.Tensor]):

        cutoff_radius = self.hypers["cutoff radius"]
        cartesian_vectors = get_cartesian_vectors(structures, cutoff_radius)

        bare_cartesian_vectors = cartesian_vectors.values.squeeze(dim=-1)
        r = torch.sqrt(
            (bare_cartesian_vectors**2)
            .sum(dim=-1)
        )
        samples_metadata = cartesian_vectors.samples  # This can be needed by the radial basis to do alchemical contractions
        radial_basis = self.radial_basis_calculator(r, samples_metadata)

        spherical_harmonics = self.spherical_harmonics_calculator.compute(bare_cartesian_vectors)  # Get the spherical harmonics
        spherical_harmonics = torch.split(spherical_harmonics, self.spherical_harmonics_split_list, dim=1)  # Split them into l chunks

        # Use broadcasting semantics to get the products in equistore shape
        vector_expansion_blocks = []
        for l, (radial_basis_l, spherical_harmonics_l) in enumerate(zip(radial_basis, spherical_harmonics)):
            if self.is_alchemical:  # If the model is alchemical, the radial basis has one extra dimension (alpha_j)
                vector_expansion_l = radial_basis_l[:, None, :, :] * spherical_harmonics_l[:, :, None, None]
                n_max_l = vector_expansion_l.shape[3]
            else:
                vector_expansion_l = radial_basis_l[:, None, :] * spherical_harmonics_l[:, :, None]
                n_max_l = vector_expansion_l.shape[2]
            if self.is_alchemical:
                properties = Labels(
                    names = ["alpha_j", "n"],
                    values = np.stack(
                        [
                            np.repeat(-np.arange(self.n_pseudo_species), n_max_l),
                            np.tile(np.arange(n_max_l), self.n_pseudo_species)
                        ],
                        axis=1
                    )
                )
            else:
                properties = Labels.range("n", n_max_l)
            vector_expansion_blocks.append(
                TensorBlock(
                    values = vector_expansion_l.reshape(vector_expansion_l.shape[0], 2*l+1, -1),
                    samples = cartesian_vectors.samples,
                    components = [Labels(
                        names = ("m",),
                        values = np.arange(-l, l+1, dtype=np.int32).reshape(2*l+1, 1)
                    )],
                    properties = properties
                )
            )

        l_max = len(vector_expansion_blocks) - 1
        vector_expansion_tmap = TensorMap(
            keys = Labels(
                names = ("l",),
                values = np.arange(0, l_max+1, dtype=np.int32).reshape(l_max+1, 1),
            ),
            blocks = vector_expansion_blocks
        )

        return vector_expansion_tmap


def get_cartesian_vectors(structures: Dict[str, torch.Tensor], cutoff_radius: float):

    labels = []
    vectors = []

    for structure_index in range(structures["n_structures"]):

        where_selected_structure = np.where(structures["structure_indices"] == structure_index)[0]

        centers, neighbors, unit_cell_shift_vectors = get_neighbor_list(
            structures["positions"].detach().cpu().numpy()[where_selected_structure], 
            structures["pbcs"][structure_index], 
            structures["cells"][structure_index], 
            cutoff_radius) 
        
        atoms_idx = torch.LongTensor(where_selected_structure)
        positions = structures["positions"][atoms_idx]
        cell = torch.tensor(np.array(structures["cells"][structure_index]), dtype=torch.get_default_dtype())
        species = structures["atomic_species"][atoms_idx]

        structure_vectors = positions[neighbors] - positions[centers] + (unit_cell_shift_vectors @ cell).to(positions.device)  # Warning: it works but in a weird way when there is no cell
        vectors.append(structure_vectors)
        labels.append(
            np.stack([
                np.array([structure_index]*len(centers)), 
                centers.numpy(), 
                neighbors.numpy(), 
                species[centers], 
                species[neighbors],
                unit_cell_shift_vectors[:, 0],
                unit_cell_shift_vectors[:, 1],
                unit_cell_shift_vectors[:, 2]
            ], axis=-1))

    vectors = torch.cat(vectors, dim=0)
    labels = np.concatenate(labels, axis=0)
    
    block = TensorBlock(
        values = vectors.unsqueeze(dim=-1),
        samples = Labels(
            names = ["structure", "center", "neighbor", "species_center", "species_neighbor", "cell_x", "cell_y", "cell_z"],
            values = np.array(labels, dtype=np.int32)
        ),
        components = [
            Labels(
                names = ["cartesian_dimension"],
                values = np.array([-1, 0, 1], dtype=np.int32).reshape((-1, 1))
            )
        ],
        properties = Labels.single()
    )

    return block
