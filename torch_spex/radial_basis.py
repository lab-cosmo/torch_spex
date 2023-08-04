import numpy as np
import torch
import equistore
from .le import get_le_spliner
from .physical_le import get_physical_le_spliner
from .normalize import normalize_true, normalize_false


class RadialBasis(torch.nn.Module):

    def __init__(self, hypers, all_species, device) -> None:
        super().__init__()

        if hypers["normalize"]:
            normalize = normalize_true
        else:
            normalize = normalize_false

        if hypers["type"] == "le":
            self.n_max_l, self.spliner = get_le_spliner(hypers["E_max"], hypers["r_cut"], hypers["normalize"], device=device)
        elif hypers["type"] == "physical":
            self.n_max_l, self.spliner = get_physical_le_spliner(hypers["E_max"], hypers["r_cut"], hypers["scale"], hypers["normalize"], hypers["cost_trade_off"], device=device)
        else:
            raise ValueError("unsupported radial basis")
        self.n_max_l = list(self.n_max_l)
        self.l_max = len(self.n_max_l) - 1
        if "alchemical" in hypers:
            self.is_alchemical = True
            self.n_pseudo_species = hypers["alchemical"]
            self.combination_matrix = normalize("embedding", torch.nn.Linear(all_species.shape[0], self.n_pseudo_species, bias=False))
            self.all_species_labels = equistore.Labels(
                names = ["species_neighbor"],
                values = all_species[:, None]
            )
        else:
            self.is_alchemical = False
        
        self.apply_mlp = False
        if hypers["mlp"]:
            self.apply_mlp = True
            self.all_species_names = range(self.n_pseudo_species) if "alchemical" in hypers else all_species
            self.radial_mlps = torch.nn.ModuleDict({
                str(l)+"_"+str(aj) : torch.nn.Sequential(
                    normalize("linear_no_bias", torch.nn.Linear(self.n_max_l[l], 32, bias=False)),
                    normalize("activation", torch.nn.SiLU()),
                    normalize("linear_no_bias", torch.nn.Linear(32, 32, bias=False)),
                    normalize("activation", torch.nn.SiLU()),
                    normalize("linear_no_bias", torch.nn.Linear(32, 32, bias=False)),
                    normalize("activation", torch.nn.SiLU()),
                    normalize("linear_no_bias", torch.nn.Linear(32, self.n_max_l[l], bias=False))
                ) for aj in self.all_species_names for l in range(self.l_max+1)
            })
        self.split_dimension = 2 if self.is_alchemical else 1

    def radial_transform(self, r):
        return r

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
            radial_functions = radial_functions.unsqueeze(1)*pseudo_species_weights.unsqueeze(2)
            # Note: if the model is alchemical, now the radial basis has one extra dimension: the alpha_j dimension, which is in the middle

        radial_basis = torch.split(radial_functions, self.n_max_l, dim=self.split_dimension)

        if self.apply_mlp:
            radial_basis_mlp = []
            if self.is_alchemical:
                for l in range(self.l_max+1):
                    radial_basis_mlp_l = []
                    for aj in self.all_species_names:
                        radial_basis_mlp_l_aj = self.radial_mlps[str(l)+"_"+str(aj)](radial_basis[l][:, aj, :])
                        radial_basis_mlp_l.append(radial_basis_mlp_l_aj)
                    radial_basis_mlp_l = torch.stack(radial_basis_mlp_l, dim=1)
                    radial_basis_mlp.append(radial_basis_mlp_l)
            else:
                for l in range(self.l_max+1):
                    neighbor_species = samples_metadata["species_neighbor"]
                    radial_basis_mlp_l = torch.zeros_like(radial_basis[l])
                    for aj in self.all_species_names:
                        where_aj = torch.tensor(np.nonzero(neighbor_species == aj)[0], dtype=torch.int64, device=radial_functions.device)
                        radial_basis_mlp_l[where_aj, :] = self.radial_mlps[str(l)+"_"+str(aj)](torch.index_select(radial_basis[l], 0, where_aj))
                    radial_basis_mlp.append(radial_basis_mlp_l)
            return radial_basis_mlp
        else:
            return radial_basis


