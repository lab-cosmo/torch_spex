import numpy as np
import torch
import ase
from torch_spex.spherical_expansions import SphericalExpansion
from torch_spex.structures import Structures

torch.set_default_dtype(torch.float64)

def get_dummy_structures(r_array):
    dummy_structures = []
    for r in r_array:
        dummy_structures.append(
            ase.Atoms('CH', positions=[(0, 0, 0), (0, 0, r)])
        )
    return dummy_structures 

# Create a fake list of dummy structures to test the radial functions generated from rascaline.

a = 6.0

r = np.linspace(0.1, a-0.001, 1000)
structures = get_dummy_structures(r)

hypers_spherical_expansion = {
    "cutoff radius": 6.0,
    "radial basis": {
        "r_cut": 6.0,
        "E_max": 200 
    }
}

calculator = SphericalExpansion(hypers_spherical_expansion, [1, 6])
spherical_expansion_coefficients = calculator(Structures(structures))

block_C_0 = spherical_expansion_coefficients.block(a_i = 6, lam = 0)
print("Block shape is", block_C_0.values.shape)

block_C_0_0 = block_C_0.values[:, :, 2].flatten()
spherical_harmonics_0 = 1.0/np.sqrt(4.0*np.pi)

all_species = np.unique(spherical_expansion_coefficients.keys["a_i"])

import matplotlib.pyplot as plt
plt.plot(r, block_C_0_0/spherical_harmonics_0, label="output")  # rascaline bug?
plt.plot([0.0, a], [0.0, 0.0], "black")
# plt.plot(r, function_for_splining(n=6, l=0, x=r), "--", label="original function")
plt.xlim(0.0, a)
plt.legend()
plt.savefig("radial.pdf")
