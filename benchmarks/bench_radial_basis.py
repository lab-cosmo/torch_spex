# PR COMMENT: for debug purposes, will be removed before merge
#print("PR DEBUG INFO: os.getcwd()", os.getcwd(), flush=True)
#print("PR DEBUG INFO: os.environ", os.environ, flush=True)

from .datasets import AlchemicalDataset

import copy
import numpy as np
import torch

import os


class RadialBasisSuite:
    #timeout = 600 # PR COMMENT debug

    alchemical_dataset_float32_cpu = AlchemicalDataset(dtype=torch.float32, device="cpu")
    alchemical_dataset_float64_cpu = AlchemicalDataset(dtype=torch.float64, device="cpu")
    # PR COMMENT later this should be 
    #alchemical_dataset_float64_cpu = alchemical_dataset_float32_cpu.to(dtype=torch.float64)
    # TODO check if gpu is available and do gpu benchs 

    def setup(self):
        self.alchemical_dataset_float32_cpu.setup_input()
        self.alchemical_dataset_float64_cpu.setup_input()

    def time_radial_basis_on_alchemical_dataset_float32_cpu(self):
        alchemical_dataset_float32_cpu.radial_basis_calculator(
                alchemical_dataset_float32_cpu.r,
                alchemical_dataset_float32_cpu.samples_metadata
        )

    def time_radial_basis_on_alchemical_dataset_float64_cpu(self):
        alchemical_dataset_float64_cpu.radial_basis_calculator(
                alchemical_dataset_float64_cpu.r,
                alchemical_dataset_float64_cpu.samples_metadata
        )

