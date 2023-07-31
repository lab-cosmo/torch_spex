import numpy as np
import torch
import equistore
from .le import get_le_spliner
from .normalize import normalize_true, normalize_false


class RadialBasis(torch.nn.Module):

    def __init__(self, hypers, all_species, device) -> None:
        super().__init__()

        if hypers["normalize"]:
            normalize = normalize_true
        else:
            normalize = normalize_false

        self.n_max_l, self.spliner = get_le_spliner(hypers["E_max"], hypers["r_cut"], device=device)
        self.l_max = len(self.n_max_l) - 1
        self.radial_transform = (lambda x: x)
        if "alchemical" in hypers:
            self.is_alchemical = True
            self.n_pseudo_species = hypers["alchemical"]
            self.combination_matrix = normalize(torch.nn.Linear(all_species.shape[0], self.n_pseudo_species, bias=False))
            self.all_species_labels = equistore.Labels(
                names = ["species_neighbor"],
                values = all_species[:, None]
            )
        else:
            self.is_alchemical = False
        
        all_species_names = range(self.n_pseudo_species) if "alchemical" in hypers else all_species
        self.radial_mlps = torch.nn.ModuleDict({
            str(l)+"_"+str(aj) : torch.nn.Sequential(
                normalize(torch.nn.Linear(self.n_max_l[l], 32, bias=False)),
                normalize(torch.nn.SiLU()),
                normalize(torch.nn.Linear(32, 32, bias=False)),
                normalize(torch.nn.SiLU()),
                normalize(torch.nn.Linear(32, 32, bias=False)),
                normalize(torch.nn.SiLU()),
                normalize(torch.nn.Linear(32, self.n_max_l[l], bias=False))
            ) for aj in range(all_species_names) for l in range(self.l_max+1)
        })

    def forward(self, r, samples_metadata):

        x = self.radial_transform(r)
        radial_functions = self.spliner.compute(x)

        if self.is_alchemical:
            one_hot_aj = torch.tensor(
                equistore.one_hot(samples_metadata, self.all_species_labels),
                dtype = torch.get_default_dtype(),
                device = radial_functions.device
            )
            pseudo_species_weights = self.combination_matrix(one_hot_aj)

        radial_basis = []
        index = 0
        for l in range(self.l_max+1):
            radial_basis_l = radial_functions[:, index:index+self.n_max_l[l]]
            if self.is_alchemical:
                radial_basis_l = radial_basis_l[:, None, :] * pseudo_species_weights[:, :, None]
                # Note: if the model is alchemical, now the radial basis has one extra dimension: the alpha_j dimension, which is in the middle
            radial_basis.append(radial_basis_l)
            index += self.n_max_l[l]

        for l in range(self.l_max+1):
            for aj in range(self.n_pseudo_species):
                radial_basis[l][:, aj, :] = self.radial_mlps[str(l)+"_"+str(aj)](radial_basis[l][:, aj, :].clone())

        # NEED APPLY OPTION FOR NON-ALCHEMICAL MODEL

        return radial_basis


