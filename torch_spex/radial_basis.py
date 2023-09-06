import torch
import metatensor.torch
from metatensor.torch import Labels
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
        elif hypers["type"] == "custom":
            # The custom keyword here allows the user to set splines from outside.
            # After initialization of the model, the user will have to use generate_splines()
            # from splines.py to generate splines and set these attributes (which we initialize to
            # None for now) from outside
            self.n_max_l, self.spliner = None, None
        else:
            raise ValueError(f"unsupported radial basis type {hypers['type']!r}")
        
        self.all_species = all_species
        self.n_max_l = list(self.n_max_l)
        self.l_max = len(self.n_max_l) - 1
        if "alchemical" in hypers:
            self.is_alchemical = True
            self.n_pseudo_species = hypers["alchemical"]
            self.combination_matrix = normalize("embedding", torch.nn.Linear(len(all_species), self.n_pseudo_species, bias=False))
            self.species_neighbor_labels = Labels(
                names = ["species_neighbor"],
                values = torch.tensor(self.all_species, dtype=torch.int).unsqueeze(1)
            )
        else:
            self.is_alchemical = False
            self.n_pseudo_species = 0  # dummy for torchscript
            self.combination_matrix = torch.nn.Linear(0, 0)  # dummy for torchscript
            self.species_neighbor_labels = Labels.empty("dummy")
        
        self.apply_mlp = False
        if hypers["mlp"]:
            self.apply_mlp = True
            # The pseudo-species, if present, are referred as 0, 1, 2, 3,...
            self.all_species_names = list(range(self.n_pseudo_species)) if "alchemical" in hypers else all_species
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

    def forward(self, r, samples_metadata: Labels):

        x = self.radial_transform(r)
        radial_functions = self.spliner.compute(x)

        if self.is_alchemical:
            one_hot_aj = metatensor.torch.one_hot(
                samples_metadata,
                self.species_neighbor_labels.to(samples_metadata.values.device)
            )
            pseudo_species_weights = self.combination_matrix(one_hot_aj.to(dtype=radial_functions.dtype))
            radial_functions = radial_functions.unsqueeze(1)*pseudo_species_weights.unsqueeze(2)
            # Note: if the model is alchemical, now the radial basis has one extra dimension: the alpha_j dimension, which is in the middle

        radial_basis = torch.split(radial_functions, self.n_max_l, dim=self.split_dimension)

        if self.apply_mlp:
            radial_basis_after_mlp = [torch.zeros_like(radial_basis[l]) for l in range(self.l_max+1)]
            if self.is_alchemical:
                for l_alphaj, radial_mlp_l_alphaj in self.radial_mlps.items():
                    split_l_alphaj = l_alphaj.split("_")
                    l = int(split_l_alphaj[0])
                    alphaj = int(split_l_alphaj[1])
                    radial_basis_after_mlp[l][:, alphaj, :] = radial_mlp_l_alphaj(radial_basis[l][:, alphaj, :])
            else:
                radial_basis_after_mlp = [torch.zeros_like(radial_basis[l]) for l in range(self.l_max+1)]
                neighbor_species = samples_metadata.column("species_neighbor")
                for l_aj, radial_mlp_l_aj in self.radial_mlps.items():
                    split_l_aj = l_aj.split("_")
                    l = int(split_l_aj[0])
                    aj = int(split_l_aj[1])
                    where_aj = torch.nonzero(neighbor_species == aj)[0]
                    radial_basis_after_mlp[l][where_aj, :] = radial_mlp_l_aj(torch.index_select(radial_basis[l], 0, where_aj))
            return radial_basis_after_mlp
        else:
            return radial_basis
