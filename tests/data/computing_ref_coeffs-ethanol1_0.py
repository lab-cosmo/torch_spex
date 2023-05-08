from torch_spex.spherical_expansions import VectorExpansion, SphericalExpansion
from torch_spex.structures import Structures

import numpy as np
import json
import equistore
import ase.io

frames = ase.io.read('../../datasets/rmd17/ethanol1.extxyz', ':1')
structures = Structures(frames)
all_species = np.unique(np.hstack([frame.numbers for frame in frames]))

hypers = {
    "cutoff radius": 3,
    "radial basis": {
        "r_cut": 3,
        "E_max": 30
    }
}
with open("expansion_coeffs-ethanol1_0-hypers.json", "w") as f:
    json.dump(hypers, f)

vector_expansion = VectorExpansion(hypers, device="cpu")
vexp_coeffs = vector_expansion.forward(structures)
equistore.save("vector_expansion_coeffs-ethanol1_0-data.npz", vexp_coeffs)

spherical_expansion_calculator = SphericalExpansion(hypers, all_species)
sexp_coeffs = spherical_expansion_calculator.forward(structures)
equistore.save("spherical_expansion_coeffs-ethanol1_0-data.npz", sexp_coeffs)
