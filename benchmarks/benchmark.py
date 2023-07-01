import time
import numpy as np
import scipy as sp
import torch
from scipy.special import spherical_jn as j_l
import ase.io
import rascaline
rascaline._c_lib._get_library()
from torch_spex.le import Jn_zeros
from equistore.torch import Labels
from torch_spex.spherical_expansions import SphericalExpansion
from torch_spex.structures import Structures

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

torch.set_default_dtype(torch.float64)

a = 6.0
E_max = 200

structures = ase.io.read("../datasets/rmd17/ethanol1.extxyz", ":20")

hypers_spherical_expansion = {
    "cutoff radius": 6.0,
    "radial basis": {
        "r_cut": 6.0,
        "E_max": 200 
    }
}
calculator = SphericalExpansion(hypers_spherical_expansion, [1, 6, 8], device=device)
transformed_structures = Structures(structures)
transformed_structures.to(device)

from torch.profiler import profile

start_time = time.time()
if True: #with profile() as prof:
    for _ in range(100):
        spherical_expansion_coefficients_torch_spex = calculator(transformed_structures)
finish_time = time.time()
print(f"torch_spex took {finish_time-start_time} s")

all_species = np.unique(spherical_expansion_coefficients_torch_spex.keys["a_i"])

l_max = 0
for key, block in spherical_expansion_coefficients_torch_spex.items():
    l_max = max(l_max, key[1])
print("l_max is", l_max)
n_max = spherical_expansion_coefficients_torch_spex.block(0).values.shape[2] // len(all_species)
print("n_max is", n_max)

l_big = 50
n_big = 50

z_ln = Jn_zeros(l_big, n_big)  # Spherical Bessel zeros
z_nl = z_ln.T

def R_nl(n, l, r):
    return j_l(l, z_nl[n, l]*r/a)

def N_nl(n, l):
    # Normalization factor for LE basis functions
    def function_to_integrate_to_get_normalization_factor(x):
        return j_l(l, x)**2 * x**2
    # print(f"Trying to integrate n={n} l={l}")
    integral, _ = sp.integrate.quadrature(function_to_integrate_to_get_normalization_factor, 0.0, z_nl[n, l], maxiter=100)
    return (1.0/z_nl[n, l]**3 * integral)**(-0.5)

def get_LE_function(n, l, r):
    R = np.zeros_like(r)
    for i in range(r.shape[0]):
        R[i] = R_nl(n, l, r[i])
    return N_nl(n, l)*R*a**(-1.5)  # This is what makes the results different when you increasing a indefinitely.
    '''
    # second kind
    ret = y_l(l, z_nl[n, l]*r/a)
    for i in range(len(ret)):
        if ret[i] < -1000000000.0: ret[i] = -1000000000.0

    return ret
    '''

def cutoff_function(r):
    cutoff = 3.0
    width = 0.5
    ret = np.zeros_like(r)
    for i, single_r in enumerate(r):
        ret[i] = (0.5*(1.0+np.cos(np.pi*(single_r-cutoff+width)/width)) if single_r > cutoff-width else 1.0)
    return ret

def radial_scaling(r):
    rate = 1.0
    scale = 2.0
    exponent = 7.0
    return rate / (rate + (r / scale) ** exponent)

def get_LE_radial_scaling(n, l, r):
    return get_LE_function(n, l, r)*radial_scaling(r)*cutoff_function(r)

def function_for_splining(n, l, x):
    return get_LE_function(n, l, x)

def function_for_splining_derivative(n, l, r):
    delta = 1e-6
    all_derivatives_except_first_and_last = (function_for_splining(n, l, r[1:-1]+delta) - function_for_splining(n, l, r[1:-1]-delta)) / (2.0*delta)
    derivative_at_zero = (function_for_splining(n, l, np.array([delta/10.0])) - function_for_splining(n, l, np.array([0.0]))) / (delta/10.0)
    derivative_last = (function_for_splining(n, l, np.array([a])) - function_for_splining(n, l, np.array([a-delta/10.0]))) / (delta/10.0)
    return np.concatenate([derivative_at_zero, all_derivatives_except_first_and_last, derivative_last])

spline_points = rascaline.generate_splines(
    function_for_splining,
    function_for_splining_derivative,
    n_max,
    l_max,
    a
)

hypers_rascaline = {
    "cutoff": a,
    "max_radial": int(n_max),
    "max_angular": int(l_max),
    "center_atom_weight": 0.0,
    "radial_basis": {"TabulatedRadialIntegral": {"points": spline_points}},
    "atomic_gaussian_width": 100.0,
    "cutoff_function": {"Step": {}},
}

calculator = rascaline.SphericalExpansion(**hypers_rascaline)

start_time = time.time()
for _ in range(100):
    spherical_expansion_coefficients_rascaline = calculator.compute(structures)
finish_time = time.time()
print(f"Rascaline took {finish_time-start_time} s")

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
