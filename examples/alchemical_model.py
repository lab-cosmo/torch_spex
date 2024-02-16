import numpy as np
import torch
from dataset import get_dataset_slices
from torch_spex.forces import compute_forces
from torch_spex.structures import InMemoryDataset, TransformerNeighborList, TransformerProperty, collate_nl
from torch_spex.spherical_expansions import SphericalExpansion
from torch_spex.atomic_composition import AtomicComposition
from power_spectrum import PowerSpectrum
from torch_spex.normalize import get_average_number_of_neighbors, normalize_true, normalize_false
from torch_spex.normalize import get_2_mom

from typing import Dict
from metatensor.torch import TensorMap

# Conversions

def get_conversions():
    
    conversions = {}
    conversions["HARTREE_TO_EV"] = 27.211386245988
    conversions["HARTREE_TO_KCAL_MOL"] = 627.509608030593
    conversions["EV_TO_KCAL_MOL"] = conversions["HARTREE_TO_KCAL_MOL"]/conversions["HARTREE_TO_EV"]
    conversions["KCAL_MOL_TO_MEV"] = 0.0433641153087705*1000.0
    conversions["METHANE_FORCE"] = conversions["HARTREE_TO_KCAL_MOL"]/0.529177
    conversions["NO_CONVERSION"] = 1.0

    return conversions

# Error measures

def get_mae(first, second):
    return torch.mean(torch.abs(first - second))

def get_rmse(first, second):
    return torch.sqrt(torch.mean((first - second)**2))

def get_sse(first, second):
    return torch.sum((first - second)**2)

torch.set_default_dtype(torch.float64)

print("DESCRIPTION")

# Unpack options
random_seed = 123123
energy_conversion = "NO_CONVERSION"
force_conversion = "NO_CONVERSION"
target_key = "energy"
dataset_path = "../datasets/alchemical.xyz"
do_forces = True
force_weight = 10.0
n_test = 200
n_train = 200
r_cut = 6.0
optimizer_name = "Adam"

np.random.seed(random_seed)
torch.manual_seed(random_seed)
print(f"Random seed: {random_seed}")

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Training on {device}")

conversions = get_conversions()
energy_conversion_factor = conversions[energy_conversion]
if do_forces:
    force_conversion_factor = conversions[force_conversion]

if "rmd17" in dataset_path:
    train_slice = str(0) + ":" + str(n_train)
    test_slice = str(0) + ":" + str(n_test)
else:
    test_slice = str(0) + ":" + str(n_test)
    train_slice = str(n_test) + ":" + str(n_test+n_train)

train_structures, test_structures = get_dataset_slices(dataset_path, train_slice, test_slice)

n_pseudo = 4
normalize = True
print("normalize", normalize)
hypers = {
    "alchemical": n_pseudo,
    "cutoff radius": r_cut,
    "radial basis": {
        "mlp": True,
        "type": "physical",
        "scale": 3.0,
        "E_max": 350,
        "normalize": True,
        "cost_trade_off": False
    }
}
if not normalize:
    normalize_func = normalize_false
else:
    hypers["normalize"] = get_average_number_of_neighbors(train_structures, r_cut)
    print(hypers["normalize"])
    normalize_func = normalize_true

average_number_of_atoms = sum([structure.get_atomic_numbers().shape[0] for structure in train_structures])/len(train_structures)
print("Average number of atoms per structure:", average_number_of_atoms)

all_species = np.sort(np.unique(np.concatenate([train_structure.numbers for train_structure in train_structures] + [test_structure.numbers for test_structure in test_structures])))
all_species = [int(species) for species in all_species]  # convert to Python ints for tracer
print(f"All species: {all_species}")


class Model(torch.nn.Module):

    def __init__(self, hypers, all_species, do_forces) -> None:
        super().__init__()
        self.all_species = all_species
        self.spherical_expansion_calculator = SphericalExpansion(hypers, all_species, device=device)
        n_max = self.spherical_expansion_calculator.vector_expansion_calculator.radial_basis_calculator.n_max_l
        print("Radial basis:", n_max)
        l_max = len(n_max) - 1
        n_feat = sum([n_max[l]**2 * n_pseudo**2 for l in range(l_max+1)])
        self.ps_calculator = PowerSpectrum(l_max, all_species)
        """
        self.nu2_model = torch.nn.ModuleDict({
            str(a_i): torch.nn.Linear(n_feat, 1, bias=False) for a_i in self.all_species
        })
        """
        self.nu2_model = torch.nn.ModuleDict({
            str(a_i): torch.nn.Sequential(
                normalize_func("linear_no_bias", torch.nn.Linear(n_feat, 256, bias=False)),
                normalize_func("activation", torch.nn.SiLU()),
                normalize_func("linear_no_bias", torch.nn.Linear(256, 256, bias=False)),
                normalize_func("activation", torch.nn.SiLU()),
                normalize_func("linear_no_bias", torch.nn.Linear(256, 256, bias=False)),
                normalize_func("activation", torch.nn.SiLU()),
                normalize_func("linear_no_bias", torch.nn.Linear(256, 1, bias=False))
            ) for a_i in self.all_species
        })
        # """
        self.comp_calculator = AtomicComposition(all_species)
        self.composition_coefficients = None  # Needs to be set from outside
        self.do_forces = do_forces
        self.normalize = normalize
        self.average_number_of_atoms = average_number_of_atoms

    def forward(self, structure_batch: Dict[str, torch.Tensor], is_training: bool = True):

        n_structures = structure_batch["cells"].shape[0]
        energies = torch.zeros(
            (n_structures,),
            dtype=structure_batch["positions"].dtype,
            device=structure_batch["positions"].device,
        )

        if self.do_forces:
            structure_batch["positions"].requires_grad_(True)

        # print("Calculating spherical expansion")
        spherical_expansion = self.spherical_expansion_calculator(
            positions = structure_batch["positions"],
            cells = structure_batch["cells"],
            species = structure_batch["species"],
            cell_shifts = structure_batch["cell_shifts"],
            centers = structure_batch["centers"],
            pairs = structure_batch["pairs"],
            structure_centers = structure_batch["structure_centers"],
            structure_pairs = structure_batch["structure_pairs"],
            structure_offsets = structure_batch["structure_offsets"]
        )
        ps = self.ps_calculator(spherical_expansion)

        # print("Calculating energies")
        atomic_energies = []
        structure_indices = []
        for ai, layer_ai in self.nu2_model.items():
            block = ps.block({"center_type": int(ai)})
            # print(block.values)
            features = block.values.squeeze(dim=1)
            structure_indices.append(block.samples.column("structure"))
            atomic_energies.append(
                layer_ai(features).squeeze(dim=-1)
            )
        atomic_energies = torch.concat(atomic_energies)
        structure_indices = torch.concatenate(structure_indices)
        # print("Before aggregation", torch.mean(atomic_energies), get_2_mom(atomic_energies))
        energies.index_add_(dim=0, index=structure_indices, source=atomic_energies)
        if self.normalize: energies = energies * self.average_number_of_atoms**(-0.5)

        comp = self.comp_calculator(
            positions = structure_batch["positions"],
            cells = structure_batch["cells"],
            species = structure_batch["species"],
            cell_shifts = structure_batch["cell_shifts"],
            centers = structure_batch["centers"],
            pairs = structure_batch["pairs"],
            structure_centers = structure_batch["structure_centers"],
            structure_pairs = structure_batch["structure_pairs"],
            structure_offsets = structure_batch["structure_offsets"]
        )
        energies += comp @ self.composition_coefficients

        # print("Computing forces by backpropagation")
        if self.do_forces:
            forces = compute_forces(energies, structure_batch["positions"], is_training=is_training)
        else:
            forces = None  # Or zero-dimensional tensor?

        return energies, forces


def predict_epoch(model, data_loader):
    
    predicted_energies = []
    predicted_forces = []
    for batch in data_loader:
        batch.pop("energies")
        batch.pop("forces")
        predicted_energies_batch, predicted_forces_batch = model(batch, is_training=False)
        predicted_energies.append(predicted_energies_batch)
        predicted_forces.append(predicted_forces_batch)

    predicted_energies = torch.concatenate(predicted_energies, dim=0)
    predicted_forces = torch.concatenate(predicted_forces, dim=0)
    return predicted_energies, predicted_forces


def train_epoch(model, data_loader, force_weight):
    
    if optimizer_name == "Adam":
        total_loss = 0.0
        for batch in data_loader:
            energies = batch.pop("energies")
            forces = batch.pop("forces")
            optimizer.zero_grad()
            predicted_energies, predicted_forces = model(batch)

            loss = get_sse(predicted_energies, energies)
            if do_forces:
                forces = forces.to(device)
                loss += force_weight * get_sse(predicted_forces, forces)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    else:
        def closure():
            optimizer.zero_grad()
            total_loss = 0.0
            for batch in data_loader:
                energies = batch.pop("energies")
                forces = batch.pop("forces")
                predicted_energies, predicted_forces = model(batch)

                loss = get_sse(predicted_energies, energies)
                if do_forces:
                    forces = forces.to(device)
                    loss += force_weight * get_sse(predicted_forces, forces)
                loss.backward()
                total_loss += loss.item()
            print(total_loss)
            return total_loss

        total_loss = optimizer.step(closure)
    return total_loss


if optimizer_name == "Adam":
    batch_size = 8  # Batch for training speed
else:
    batch_size = 128  # Batch for memory

print("Precomputing neighborlists")

transformers = [
    TransformerNeighborList(cutoff=hypers["cutoff radius"], device=device),
    TransformerProperty("energies", lambda frame: torch.tensor([frame.info["energy"]], dtype=torch.get_default_dtype(), device=device)*energy_conversion_factor),
]
if do_forces: transformers.append(TransformerProperty("forces", lambda frame: torch.tensor(frame.get_forces(), dtype=torch.get_default_dtype(), device=device)*force_conversion_factor))

predict_train_dataset = InMemoryDataset(train_structures, transformers)
predict_test_dataset = InMemoryDataset(test_structures, transformers)
train_dataset = InMemoryDataset(train_structures, transformers)  # avoid sharing tensors between different dataloaders

predict_train_data_loader = torch.utils.data.DataLoader(predict_train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_nl)
predict_test_data_loader = torch.utils.data.DataLoader(predict_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_nl)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_nl)

print("Finished neighborlists")


print("Linear fit for one-body energies")

train_energies = torch.tensor([structure.info[target_key] for structure in train_structures])*energy_conversion_factor
train_energies = train_energies.to(device)
test_energies = torch.tensor([structure.info[target_key] for structure in test_structures])*energy_conversion_factor
test_energies = test_energies.to(device)

if do_forces:
    train_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in train_structures], axis=0))*force_conversion_factor
    train_forces = train_forces.to(device)
    test_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in test_structures], axis=0))*force_conversion_factor
    test_forces = test_forces.to(device)

comp_calculator = AtomicComposition(all_species)
train_comp = []
for batch in predict_train_data_loader:
    batch.pop("energies")
    batch.pop("forces")
    train_comp.append(
        comp_calculator(**batch)
    )
train_comp = torch.concatenate(train_comp)
c_comp = torch.linalg.solve(train_comp.T @ train_comp, train_comp.T @ train_energies)

model = Model(hypers, all_species, do_forces=do_forces).to(device)
model.composition_coefficients = c_comp

# Deactivate kernel fusion which slows down the model.
# With kernel fusion, our model would be recompiled at every call
# due to the varying shapes of the involved tensors (neighborlists 
# can vary between different structures and batches)
# Perhaps [("DYNAMIC", 1)] can offer better performance
torch.jit.set_fusion_strategy([("DYNAMIC", 0)])
model = torch.jit.script(model)

if optimizer_name == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
else:
    optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe", history_size=128)


print("Finished linear fit for one-body energies")


"""
# cProfile run:
import cProfile
cProfile.runctx('model.train_epoch(train_data_loader, force_weight)', globals(), locals(), 'profile')

import pstats
stats = pstats.Stats('profile')
stats.strip_dirs().sort_stats('tottime').print_stats(100)

import os
os.remove('profile')
"""
"""
from torch.profiler import profile

with profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as prof:
"""

for epoch in range(1000):

    # print(torch.cuda.max_memory_allocated())

    predicted_train_energies, predicted_train_forces = predict_epoch(model, predict_train_data_loader)
    predicted_test_energies, predicted_test_forces = predict_epoch(model, predict_test_data_loader)

    print()
    print(f"Epoch number {epoch}, Total loss: {get_sse(predicted_train_energies, train_energies)+force_weight*get_sse(predicted_train_forces, train_forces)}")
    print(f"Energy errors: Train RMSE: {get_rmse(predicted_train_energies, train_energies)}, Train MAE: {get_mae(predicted_train_energies, train_energies)}, Test RMSE: {get_rmse(predicted_test_energies, test_energies)}, Test MAE: {get_mae(predicted_test_energies, test_energies)}")
    if do_forces:
        print(f"Force errors: Train RMSE: {get_rmse(predicted_train_forces, train_forces)}, Train MAE: {get_mae(predicted_train_forces, train_forces)}, Test RMSE: {get_rmse(predicted_test_forces, test_forces)}, Test MAE: {get_mae(predicted_test_forces, test_forces)}")

    _ = train_epoch(model, train_data_loader, force_weight)

#print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
