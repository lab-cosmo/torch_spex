import numpy as np
import torch
from dataset import get_dataset_slices
from torch_spex.forces import compute_forces
from torch_spex.spherical_expansions import SphericalExpansion
from power_spectrum import PowerSpectrum
from torch_spex.structures import InMemoryDataset, TransformerNeighborList, TransformerProperty, collate_nl

from typing import Dict

# Conversions

def get_conversions():
    
    conversions = {}
    conversions["HARTREE_TO_EV"] = 27.211386245988
    conversions["HARTREE_TO_KCAL_MOL"] = 627.509608030593
    conversions["EV_TO_KCAL_MOL"] = conversions["HARTREE_TO_KCAL_MOL"]/conversions["HARTREE_TO_EV"]
    conversions["KCAL_MOL_TO_MEV"] = 0.0433641153087705*1000.0
    conversions["METHANE_FORCE"] = conversions["HARTREE_TO_KCAL_MOL"]/0.529177

    return conversions

# Error measures

def get_mae(first, second):
    return torch.mean(torch.abs(first - second))

def get_rmse(first, second):
    return torch.sqrt(torch.mean((first - second)**2))

def get_sse(first, second):
    return torch.sum((first - second)**2)

torch.set_default_dtype(torch.float64)

# Unpack options
random_seed = 123123
energy_conversion = "KCAL_MOL_TO_MEV"
force_conversion = "KCAL_MOL_TO_MEV"
target_key = "energy"
dataset_path = "../datasets/rmd17/ethanol1.extxyz"
do_forces = True
force_weight = 1.0
n_test = 1000
n_train = 50
r_cut = 6.0
optimizer_name = "Adam"

np.random.seed(random_seed)
torch.manual_seed(random_seed)
print(f"Random seed: {random_seed}")

device = "cuda" if torch.cuda.is_available() else "cpu"
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

hypers = {
    "cutoff radius": r_cut,
    "radial basis": {
        "mlp": True,
        "type": "physical",
        "scale": 1.7,
        "E_max": 3000,
        "normalize": True,
        "cost_trade_off": False
    }
}

all_species = np.sort(np.unique(np.concatenate([train_structure.numbers for train_structure in train_structures] + [test_structure.numbers for test_structure in test_structures])))
all_species = [int(species) for species in all_species]  # convert to Python ints for tracer
print(f"All species: {all_species}")


class Model(torch.nn.Module):

    def __init__(self, hypers, all_species, do_forces) -> None:
        super().__init__()
        self.all_species = all_species
        self.spherical_expansion_calculator = SphericalExpansion(hypers, all_species)
        n_max = self.spherical_expansion_calculator.vector_expansion_calculator.radial_basis_calculator.n_max_l
        print(n_max)
        l_max = len(n_max) - 1
        n_feat = sum([n_max[l]**2 * len(all_species)**2 for l in range(l_max+1)])
        self.ps_calculator = PowerSpectrum(l_max, all_species)
        """
        self.nu2_model = torch.nn.ModuleDict({
            str(a_i): torch.nn.Linear(n_feat, 1, bias=False) for a_i in self.all_species
        })
        """
        self.nu2_model = torch.nn.ModuleDict({
            str(a_i): torch.nn.Sequential(
                torch.nn.Linear(n_feat, 256),
                torch.nn.SiLU(),
                torch.nn.Linear(256, 256),
                torch.nn.SiLU(),
                torch.nn.Linear(256, 256),
                torch.nn.SiLU(),
                torch.nn.Linear(256, 1)
            ) for a_i in self.all_species
        })
        # """
        self.energy_shift = 0.0  # to be set from outside. Could also be a nn.Parameter
        self.do_forces = do_forces

    def forward(self, structures: Dict[str, torch.Tensor], is_training: bool = True):

        # print("Transforming structures")
        energies = torch.zeros((len(structures["cells"]),), device=structures["positions"].device, dtype=structures["positions"].dtype)

        if self.do_forces:
            structures["positions"].requires_grad_(True)

        # print("Calculating spherical expansion")
        spherical_expansion = self.spherical_expansion_calculator(
            positions = structures["positions"],
            cells = structures["cells"],
            species = structures["species"],
            cell_shifts = structures["cell_shifts"],
            centers = structures["centers"],
            pairs = structures["pairs"],
            structure_centers = structures["structure_centers"],
            structure_pairs = structures["structure_pairs"],
            structure_offsets = structures["structure_offsets"]
        )
        ps = self.ps_calculator(spherical_expansion)

        atomic_energies = []
        structure_indices = []
        for ai, model_ai in self.nu2_model.items():
            block = ps.block({"center_type": int(ai)})
            features = block.values.squeeze(dim=1)
            structure_indices.append(block.samples.column("structure"))
            atomic_energies.append(
                model_ai(features).squeeze(dim=-1)
            )
        atomic_energies = torch.concat(atomic_energies)
        structure_indices = torch.concatenate(structure_indices)
        
        energies.index_add_(dim=0, index=structure_indices, source=atomic_energies)
        energies += self.energy_shift

        # print("Computing forces by backpropagation")
        if self.do_forces:
            forces = compute_forces(energies, structures["positions"], is_training=is_training)
        else:
            forces = None  # Or zero-dimensional tensor?

        return energies, forces


def predict_epoch(model, data_loader):
    
    predicted_energies = []
    predicted_forces = []
    for batch in data_loader:
        batch.pop("energies")
        if model.do_forces: batch.pop("forces")
        predicted_energies_batch, predicted_forces_batch = model(batch, is_training=False)
        predicted_energies.append(predicted_energies_batch)
        if model.do_forces: predicted_forces.append(predicted_forces_batch)  # the predicted forces for the batch are themselves a list

    predicted_energies = torch.concatenate(predicted_energies, dim=0)
    if model.do_forces: predicted_forces = torch.concatenate(predicted_forces, dim=0)
    return predicted_energies, predicted_forces


def train_epoch(model, data_loader, force_weight):
    
    if optimizer_name == "Adam":
        total_loss = 0.0
        for batch in data_loader:
            optimizer.zero_grad()
            energies = batch.pop("energies")
            if model.do_forces: forces = batch.pop("forces")
            predicted_energies, predicted_forces = model(batch)
            loss = get_sse(predicted_energies, energies)
            if model.do_forces:
                loss += force_weight * get_sse(predicted_forces, forces)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    else:
        def closure():
            optimizer.zero_grad()
            for train_structures in data_loader:
                predicted_energies, predicted_forces = model(train_structures)
                energies = batch.pop("energies")
                if self.do_forces: forces = batch.pop("forces")
                loss = get_sse(predicted_energies, energies)
                if do_forces:
                    loss += force_weight * get_sse(predicted_forces, forces)
            loss.backward()
            print(loss.item())
            return loss
        loss = optimizer.step(closure)
        total_loss = loss.item()

    return total_loss


model = Model(hypers, all_species, do_forces=do_forces).to(device)

# Deactivate kernel fusion which slows down the model.
# With kernel fusion, our model would be recompiled at every call
# due to the varying shapes of the involved tensors (neighborlists 
# can vary between different structures and batches)
# Perhaps [("DYNAMIC", 1)] can offer better performance
torch.jit.set_fusion_strategy([("DYNAMIC", 0)])
model = torch.jit.script(model)

if optimizer_name == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
    batch_size = 16
else:
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-2)
    batch_size = n_train

    
print("Precomputing neighborlists")

transformers = [
    TransformerNeighborList(cutoff=hypers["cutoff radius"], device=device),
    TransformerProperty("energies", lambda frame: torch.tensor([frame.info["energy"]], dtype=torch.get_default_dtype(), device=device)*energy_conversion_factor),
]
if do_forces:
    transformers.append(TransformerProperty("forces", lambda frame: torch.tensor(frame.get_forces(), dtype=torch.get_default_dtype(), device=device)*force_conversion_factor))

predict_train_dataset = InMemoryDataset(train_structures, transformers)
predict_test_dataset = InMemoryDataset(test_structures, transformers)
train_dataset = InMemoryDataset(train_structures, transformers)  # avoid sharing tensors between different dataloaders

predict_train_data_loader = torch.utils.data.DataLoader(predict_train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_nl)
predict_test_data_loader = torch.utils.data.DataLoader(predict_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_nl)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_nl)

print("Finished neighborlists")


train_energies = torch.tensor([structure.info[target_key] for structure in train_structures])*energy_conversion_factor
train_energies = train_energies.to(device)
test_energies = torch.tensor([structure.info[target_key] for structure in test_structures])*energy_conversion_factor
test_energies = test_energies.to(device)
model.energy_shift = torch.mean(train_energies)
if do_forces:
    train_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in train_structures], axis=0))*force_conversion_factor
    train_forces = train_forces.to(device)
    test_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in test_structures], axis=0))*force_conversion_factor
    test_forces = test_forces.to(device)


for epoch in range(1000):

    predicted_train_energies, predicted_train_forces = predict_epoch(model, predict_train_data_loader)
    predicted_test_energies, predicted_test_forces = predict_epoch(model, predict_test_data_loader)

    print()
    if do_forces:
        print(f"Epoch number {epoch}, Total loss: {get_sse(predicted_train_energies, train_energies)+force_weight*get_sse(predicted_train_forces, train_forces)}")
    else:
        print(f"Epoch number {epoch}, Total loss: {get_sse(predicted_train_energies, train_energies)}")
    print(f"Energy errors: Train RMSE: {get_rmse(predicted_train_energies, train_energies)}, Train MAE: {get_mae(predicted_train_energies, train_energies)}, Test RMSE: {get_rmse(predicted_test_energies, test_energies)}, Test MAE: {get_mae(predicted_test_energies, test_energies)}")
    if do_forces:
        print(f"Force errors: Train RMSE: {get_rmse(predicted_train_forces, train_forces)}, Train MAE: {get_mae(predicted_train_forces, train_forces)}, Test RMSE: {get_rmse(predicted_test_forces, test_forces)}, Test MAE: {get_mae(predicted_test_forces, test_forces)}")

    _ = train_epoch(model, train_data_loader, force_weight)
