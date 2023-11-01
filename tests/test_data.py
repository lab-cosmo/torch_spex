from torch_spex.spherical_expansions import VectorExpansion
from torch_spex.structures import InMemoryDataset, TransformerNeighborList, TransformerProperty, collate_nl
import torch
from torch.utils.data import DataLoader

import metatensor

import json
import ase.io

def test_in_memory_neighbor_list():

    with open("tests/data/expansion_coeffs-ethanol1_0-hypers.json", "r") as f:
        hypers = json.load(f)

    n_structures = 2
    frames = ase.io.read('datasets/rmd17/ethanol1.extxyz', f':{n_structures}')
    transformers = [TransformerNeighborList(cutoff=3.),
            TransformerProperty("energy", lambda frame: torch.tensor([frame.get_total_energy()])),
            TransformerProperty("forces", lambda frame: torch.tensor(frame.get_forces()))]
    dataset = InMemoryDataset(frames, transformers)
    loader = DataLoader(dataset, batch_size=n_structures, collate_fn=collate_nl)
    batch = next(iter(loader))
    assert set(batch.keys()) == {'positions', 'species', 'cells', 'centers', 'pairs', 'cell_shifts', 'structure_centers', 'structure_pairs', 'structure_offsets', 'energy', 'forces'}
    assert len(batch['species']) == len(batch['centers']) == len(batch['structure_centers']) == len(batch['forces']) == len(batch['positions'])
    assert len(batch['cells']) == n_structures == len(batch['energy']) == len(batch['structure_offsets'])
    assert len(batch['pairs']) == len(batch['cell_shifts']) == len(batch['structure_pairs'])
