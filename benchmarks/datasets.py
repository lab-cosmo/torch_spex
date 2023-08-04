"""Computes parameters from datasets to prevent recomputation for each benchmark"""
import os
import pathlib
import abc
import inspect
import pickle
import sys
if (sys.version_info.major >= 3) and (sys.version_info.minor >= 8):
    from pickle import PickleBuffer

import hashlib

import ase.io

import numpy as np
import torch

from torch_spex.structures import InMemoryDataset, TransformerNeighborList, collate_nl
from torch.utils.data import DataLoader
from torch_spex.spherical_expansions import get_cartesian_vectors
from torch_spex.radial_basis import RadialBasis

from equistore import Labels

# To determine the datasets folder, we first check if it run with asv, then if
# it is run with tox env. If nothing we assume it is run from root of
# torch_spex project folder
if os.getenv("ASV_CONF_DIR") is not None:
    TORCH_SPEX_PATH = os.getenv("ASV_CONF_DIR")
elif os.getenv("TOX_WORK_DIR") is not None:
    TORCH_SPEX_PATH = os.path.join(os.getenv("TOX_WORK_DIR"), "../")
else:
    TORCH_SPEX_PATH = os.getenv("PWD")

PRECOMPUTED_INPUT_ROOT = os.path.join(TORCH_SPEX_PATH, ".precomputed_input")
pathlib.Path(PRECOMPUTED_INPUT_ROOT).mkdir(parents=False, exist_ok=True)

class Dataset(metaclass=abc.ABCMeta):

    def get_precomputed_input_path(self):
        global PRECOMPUTED_INPUT_ROOT
        # default python hash is not consistent over multiple runs so we use different
        # hash function
        dataset_id = self.get_datset_id()
        return pathlib.Path(os.path.join(PRECOMPUTED_INPUT_ROOT, str(dataset_id)+".pickle"))

    def get_datset_id(self):
        bench_variables = {key: self.__dict__[key]
                for key in sorted(self.__dict__.keys())
                    if key.startswith("bench_")}
        bench_variables_id = "BENCH_VARIABLES=" + str(bench_variables) + "\n"
        source_code_id = inspect.getsource(self.__class__.compute_input)
        dataset_id = bench_variables_id + source_code_id
        dataset_hashed_id = str(int.from_bytes(hashlib.md5(dataset_id.encode("utf8")).digest()))

    def setup_input(self):
        """
        Each dataset class creates a separate pickle file that stores the parameters
        of the dataset
        """
        precomputed_input_path = self.get_precomputed_input_path()

        self.compute_input()

        if precomputed_input_path.exists():
            with open(precomputed_input_path, "rb") as file:
                self.__dict__ = pickle.load(file).__dict__
        else:
            self.compute_input()
            with open(precomputed_input_path, "wb") as file:
                pickle.dump(self, file)
            with open(precomputed_input_path, "rb") as file:
                obj = pickle.load(file)

    def teardown_input(self):
        """
        Not supposed to be used for benchmarks since it removes the files with
        the cached input to not recompute
        """
        precomputed_input_path = self.get_precomputed_input_path()
        if precomputed_input_path.exists():
            os.remove(precomputed_input_path)

    @abc.abstractmethod
    def compute_input(self):
        pass

class AlchemicalDataset(Dataset):
    def __init__(self, dtype, device):
        if device != "cpu":
            # TODO TransformerNeighborList need dtype
            # TODO collate_fn needs device to move
            raise NotImplemented("Only cpu supported for the moment")

        # TODO IMPORTANT transparent way to tell user what member_variables are user for serialization of data object 
        self.bench_dtype = dtype
        self.bench_device = device

    def compute_input(self):
        global TORCH_SPEX_PATH

        self.hypers_radial_basis = {'E_max': 100, 'r_cut': 5.0, 'alchemical': 4}
        frames = ase.io.read(os.path.join(TORCH_SPEX_PATH, 'datasets/alchemical.xyz'), ':2')

        self.all_species = np.unique(np.hstack([frame.numbers for frame in frames]))
        transformers = [TransformerNeighborList(cutoff=self.hypers_radial_basis ["r_cut"])]
        dataset = InMemoryDataset(frames, transformers)
        loader = DataLoader(dataset, batch_size=len(frames), collate_fn=collate_nl)
        batch = next(iter(loader))
        batch.pop("positions")
        batch.pop("cell")

        cartesian_vectors = get_cartesian_vectors(**batch)
        bare_cartesian_vectors = cartesian_vectors.values.squeeze(dim=-1)

        self.r = torch.sqrt(
            (bare_cartesian_vectors**2)
            .sum(dim=-1)
        ).to(dtype=self.bench_dtype, device=self.bench_device)
        self.samples_metadata = cartesian_vectors.samples  # This can be needed by the radial basis to do alchemical contractions

        self.radial_basis_calculator = RadialBasis(
                self.hypers_radial_basis,
                self.all_species,
                device=self.bench_device)

    # we need to write our own pickle here as long as Label is not pickable 
    @classmethod
    def _from_pickle(cls, obj):
        new_obj = cls(obj['bench_dtype'], obj['bench_device'])
        new_obj.__dict__ = obj
        new_obj.__dict__["samples_metadata"] = Labels(
                names=obj["samples_metadata_names"],
                values=obj["samples_metadata_values"])
        new_obj__dict__.pop("samples_metadata_values")
        new_obj__dict__.pop("samples_metadata_names")
        return new_obj

    def __reduce__(self):
        obj = {
            "bench_dtype": self.bench_dtype,
            "bench_device": self.bench_device,
            "hypers_radial_basis": self.hypers_radial_basis,
            "all_species": self.all_species,
            "r": self.r,
            "radial_basis_calculator": self.radial_basis_calculator,
            "samples_metadata_values": self.samples_metadata.values,
            "samples_metadata_names": self.samples_metadata.names,
        }
        return self._from_pickle, (obj,)

    def to(self):
        raise NotImplemented("Not yet implemented")
