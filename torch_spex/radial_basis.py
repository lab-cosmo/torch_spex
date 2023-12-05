import torch
import metatensor.torch
from metatensor.torch import Labels
from .le import get_le_spliner
from .physical_LE import get_physical_le_spliner
from .normalize import normalize_true, normalize_false
from typing import Optional

class RadialBasis(torch.nn.Module):

    def __init__(self, hypers, all_species,
            device:Optional[torch.device] = None,
            dtype:Optional[torch.dtype] = None) -> None:
        super().__init__()

        # Only for the physical basis, but initialized for all branches
        # due to torchscript complaints
        lengthscales = torch.zeros((max(all_species)+1))
        for species in all_species:
            lengthscales[species] = 0.0
        self.lengthscales = torch.nn.Parameter(lengthscales)

        if hypers["normalize"]:
            normalize = normalize_true
        else:
            normalize = normalize_false

        self.is_physical = False

        if hypers["type"] == "le":
            self.n_max_l, self.spliner = get_le_spliner(hypers["E_max"],
                    hypers["r_cut"], hypers["normalize"], device=device, dtype=dtype)
        elif hypers["type"] == "physical":
            self.n_max_l, self.spliner = get_physical_le_spliner(hypers["E_max"], hypers["r_cut"], hypers["normalize"], device=device, dtype=dtype)
            self.is_physical = True
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
            self.combination_matrix = normalize("embedding",
                    torch.nn.Linear(len(all_species), self.n_pseudo_species, bias=False,
                        device=device, dtype=dtype))
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
        else:  # make torchscript happy
            self.apply_mlp = False
            self.all_species_names = []
            self.radial_mlps = torch.nn.ModuleDict({})
        
        self.split_dimension = 2 if self.is_alchemical else 1

    def radial_transform(self, r, samples_metadata: Labels):
        if self.is_physical:
            a_i = samples_metadata.column("species_center")
            a_j = samples_metadata.column("species_neighbor")
            x = r/(0.1+torch.exp(self.lengthscales[a_i])+torch.exp(self.lengthscales[a_j]))
            return x
        else:
            return r

    def forward(self, r, samples_metadata: Labels):

        x = self.radial_transform(r, samples_metadata)
        if self.is_physical:
            radial_functions = torch.where(
                x.unsqueeze(1) < 10.0,
                self.spliner.compute(x),
                0.0
            )
        else:
            radial_functions = self.spliner.compute(x)


        if self.is_alchemical:
            self.species_neighbor_labels.to(samples_metadata.values.device)  # Move device if needed
            one_hot_aj = metatensor.torch.one_hot(
                samples_metadata,
                self.species_neighbor_labels
            )
            pseudo_species_weights = \
                    self.combination_matrix(one_hot_aj.to(device=radial_functions.device, dtype=radial_functions.dtype))
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
