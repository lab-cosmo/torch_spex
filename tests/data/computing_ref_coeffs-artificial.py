# reference data was produced using torch_spex commit b9261bf2


from torch_spex.spherical_expansions import VectorExpansion, SphericalExpansion
from torch_spex.structures import ase_atoms_to_tensordict

import torch
import numpy as np
import json
import metatensor
import ase.io

frames = ase.io.read('../datasets/artificial.extxyz', ':')
structures = ase_atoms_to_tensordict(frames)
all_species = np.unique(np.hstack([frame.numbers for frame in frames]))

hypers = {
    "cutoff radius": 3,
    "radial basis": {
        "E_max": 40
    }
}
with open("expansion_coeffs-artificial-hypers.json", "w") as f:
    json.dump(hypers, f)

vector_expansion = VectorExpansion(hypers, all_species, device="cpu")

vexp_coeffs = vector_expansion.forward(structures)
metatensor.save("vector_expansion_coeffs-artificial-data.npz", vexp_coeffs)

spherical_expansion_calculator = SphericalExpansion(hypers, all_species)
sexp_coeffs = spherical_expansion_calculator.forward(structures)
metatensor.save("spherical_expansion_coeffs-artificial-data.npz", sexp_coeffs)

hypers["alchemical"] = 2
with open("expansion_coeffs-artificial-alchemical-hypers.json", "w") as f:
    json.dump(hypers, f)


torch.manual_seed(0) # set for the combination_matrix in SphericalExpansion
spherical_expansion_calculator = SphericalExpansion(hypers, all_species)
# some random combination matrix, it is only important that we use the same one in the tests
with torch.no_grad():
    spherical_expansion_calculator.radial_basis_calculator.combination_matrix.weight.copy_(
        torch.tensor(
            [[-0.00432252,  0.30971584, -0.47518533],
             [-0.4248946 , -0.22236897,  0.15482073]],
            dtype=torch.float32
        )
    )

with torch.no_grad():
    sexp_coeffs = spherical_expansion_calculator.forward(structures)
metatensor.save("spherical_expansion_coeffs-artificial-alchemical-seed0-data.npz", sexp_coeffs)
