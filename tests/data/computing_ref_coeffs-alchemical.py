# reference data was produced using commit c76f261364146517fe59eefb455383e124a4cab9
# computes reference points using alchemical dataset

from torch_spex.spherical_expansions import VectorExpansion, SphericalExpansion
from torch_spex.structures import ase_atoms_to_tensordict

import torch
import numpy as np
import json
import equistore
import ase.io

frames = ase.io.read('../../datasets/alchemical.xyz', ':2')
structures = ase_atoms_to_tensordict(frames)
all_species = np.unique(np.hstack([frame.numbers for frame in frames]))

hypers = {
    "cutoff radius": 4,
    "radial basis": {
        "E_max": 30
    }
}
with open("expansion_coeffs-alchemical_01-hypers.json", "w") as f:
    json.dump(hypers, f)

vector_expansion = VectorExpansion(hypers, all_species, device="cpu")
vexp_coeffs = vector_expansion.forward(structures)
equistore.save("vector_expansion_coeffs-alchemical_01-data.npz", vexp_coeffs)

spherical_expansion_calculator = SphericalExpansion(hypers, all_species)
sexp_coeffs = spherical_expansion_calculator.forward(structures)
equistore.save("spherical_expansion_coeffs-alchemical_01-data.npz", sexp_coeffs)

hypers["alchemical"] = 2
with open("expansion_coeffs-alchemical_01-alchemical-hypers.json", "w") as f:
    json.dump(hypers, f)


torch.manual_seed(0) # set for the combination_matrix in SphericalExpansion
spherical_expansion_calculator = SphericalExpansion(hypers, all_species)

with torch.no_grad():
    sexp_coeffs = spherical_expansion_calculator.forward(structures)
equistore.save("spherical_expansion_coeffs-alchemical_01-alchemical-seed0-data.npz", sexp_coeffs)
