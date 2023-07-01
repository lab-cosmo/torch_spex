import numpy as np
import torch
from dataset import get_dataset_slices
from torch_spex.forces import compute_forces
from torch_spex.structures import ase_atoms_to_tensordict
from torch_spex.spherical_expansions import SphericalExpansion
from power_spectrum import PowerSpectrum

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

# Unpack options
random_seed = 123123
energy_conversion = "NO_CONVERSION"
force_conversion = "NO_CONVERSION"
target_key = "energy"
dataset_path = "../datasets/alchemical.xyz"
do_forces = True
force_weight = 1.0
n_test = 100
n_train = 100
r_cut = 5.0
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

n_pseudo = 4
hypers = {
    "alchemical": n_pseudo,
    "cutoff radius": r_cut,
    "radial basis": {
        "r_cut": r_cut,
        "E_max": 300
    }
}

all_species = torch.tensor(np.sort(np.unique(np.concatenate([train_structure.numbers for train_structure in train_structures] + [test_structure.numbers for test_structure in test_structures]))))
print(f"All species: {all_species}")


class Model(torch.nn.Module):

    def __init__(self, hypers, all_species, do_forces) -> None:
        super().__init__()
        self.all_species = all_species
        self.spherical_expansion_calculator = SphericalExpansion(hypers, all_species, device=device)
        self.ps_calculator = PowerSpectrum(all_species)
        n_max = self.spherical_expansion_calculator.vector_expansion_calculator.radial_basis_calculator.n_max_l
        l_max = len(n_max) - 1
        n_feat = sum([n_max[l]**2 * n_pseudo**2 for l in range(l_max+1)])
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
        self.do_forces = do_forces

    def forward(self, structures, is_training=True):

        # print("Transforming structures")
        structures = ase_atoms_to_tensordict(structures, device=device)
        energies = torch.zeros((structures["n_structures"].item(),), device=device, dtype=torch.get_default_dtype())

        if self.do_forces:
            structures["positions"].requires_grad = True

        # print("Calculating spherical expansion")
        spherical_expansion = self.spherical_expansion_calculator(structures)
        ps = self.ps_calculator(spherical_expansion)

        # print("Calculating energies")
        self._apply_layer(energies, ps, self.nu2_model)

        # print("Computing forces by backpropagation")
        if self.do_forces:
            forces = compute_forces(energies, structures["positions"], is_training=is_training)
        else:
            forces = None  # Or zero-dimensional tensor?

        return energies, forces

    def predict_epoch(self, data_loader):
        
        predicted_energies = []
        predicted_forces = []
        for i, batch in enumerate(data_loader):
            predicted_energies_batch, predicted_forces_batch = model(batch, is_training=False)
            predicted_energies.append(predicted_energies_batch)
            predicted_forces.append(predicted_forces_batch)

        predicted_energies = torch.concatenate(predicted_energies, dim=0)
        predicted_forces = torch.concatenate(predicted_forces, dim=0)
        return predicted_energies, predicted_forces


    def train_epoch(self, data_loader, force_weight):
        
        if optimizer_name == "Adam":
            total_loss = 0.0
            for i, batch in enumerate(data_loader):
                #print(i)
                optimizer.zero_grad()
                predicted_energies, predicted_forces = model(batch)
                energies = torch.tensor([structure.info[target_key] for structure in batch], device=device)*energy_conversion_factor 

                comp = comp_calculator.compute(batch)
                comp = comp.keys_to_properties(center_species_labels)
                comp = torch.tensor(comp.block().values).to(device)
                energies -= comp @ c_comp

                loss = get_sse(predicted_energies, energies)
                if do_forces:
                    forces = torch.tensor(np.concatenate([structure.get_forces() for structure in batch], axis=0))*force_conversion_factor
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
                    predicted_energies, predicted_forces = model(batch)
                    energies = torch.tensor([structure.info[target_key] for structure in batch], device=device)*energy_conversion_factor
                    
                    comp = comp_calculator.compute(batch)
                    comp = comp.keys_to_properties(center_species_labels)
                    comp = torch.tensor(comp.block().values).to(device)
                    energies -= comp @ c_comp

                    loss = get_sse(predicted_energies, energies)
                    if do_forces:
                        forces = torch.tensor(np.concatenate([structure.get_forces() for structure in batch], axis=0))*force_conversion_factor
                        forces = forces.to(device)
                        loss += force_weight * get_sse(predicted_forces, forces)
                    loss.backward()
                    total_loss += loss.item()
                print(total_loss)
                return total_loss

            total_loss = optimizer.step(closure)
        return total_loss

    def _apply_layer(self, energies, tmap, layer):
        atomic_energies = []
        structure_indices = []
        for a_i in self.all_species:
            block = tmap.block(a_i=a_i)
            features = block.values.squeeze(dim=1)
            structure_indices.append(block.samples["structure"].values[:, 0])
            atomic_energies.append(
                layer[str(a_i)](features).squeeze(dim=-1)
            )
        atomic_energies = torch.concat(atomic_energies)
        structure_indices = torch.LongTensor(np.concatenate(structure_indices))
        
        energies.index_add_(dim=0, index=structure_indices.to(device), source=atomic_energies)


    # def print_state()... Would print loss, train errors, validation errors, test errors, ...

model = Model(hypers, all_species, do_forces=do_forces).to(device)
# print(model)

if optimizer_name == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
    batch_size = 16  # Batch for training speed
else:
    optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe", history_size=128)
    batch_size = 128  # Batch for memory

train_data_loader = torch.utils.data.DataLoader(train_structures, batch_size=batch_size, shuffle=True, collate_fn=(lambda x: x))

predict_train_data_loader = torch.utils.data.DataLoader(train_structures, batch_size=512, shuffle=False, collate_fn=(lambda x: x))
predict_test_data_loader = torch.utils.data.DataLoader(test_structures, batch_size=512, shuffle=False, collate_fn=(lambda x: x))

# with torch.autograd.set_detect_anomaly(True):
predicted_train_energies, predicted_train_forces = model.predict_epoch(predict_train_data_loader)
predicted_test_energies, predicted_test_forces = model.predict_epoch(predict_test_data_loader)
train_energies = torch.tensor([structure.info[target_key] for structure in train_structures])*energy_conversion_factor
train_energies = train_energies.to(device)
test_energies = torch.tensor([structure.info[target_key] for structure in test_structures])*energy_conversion_factor
test_energies = test_energies.to(device)

# Linear fit for one-body energies:
import rascaline
import equistore.torch as equistore
center_species_labels = equistore.Labels(
    names = ["species_center"],
    values = all_species.reshape(-1, 1)
)
comp_calculator = rascaline.AtomicComposition(per_structure=True)
train_comp = comp_calculator.compute(train_structures)
train_comp = train_comp.keys_to_properties(center_species_labels)
train_comp = torch.tensor(train_comp.block().values).to(device)

c_comp = torch.linalg.solve(train_comp.T @ train_comp, train_comp.T @ train_energies)

test_comp = comp_calculator.compute(test_structures)
test_comp = test_comp.keys_to_properties(center_species_labels)
test_comp = torch.tensor(test_comp.block().values).to(device)

train_energies -= train_comp @ c_comp
test_energies -= test_comp @ c_comp

if do_forces:
    train_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in train_structures], axis=0))*force_conversion_factor
    train_forces = train_forces.to(device)
    test_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in test_structures], axis=0))*force_conversion_factor
    test_forces = test_forces.to(device)
print()
print(f"Before training")
print(f"Energy errors: Train RMSE: {get_rmse(predicted_train_energies, train_energies)}, Train MAE: {get_mae(predicted_train_energies, train_energies)}, Test RMSE: {get_rmse(predicted_test_energies, test_energies)}, Test MAE: {get_mae(predicted_test_energies, test_energies)}")
if do_forces:
    print(f"Force errors: Train RMSE: {get_rmse(predicted_train_forces, train_forces)}, Train MAE: {get_mae(predicted_train_forces, train_forces)}, Test RMSE: {get_rmse(predicted_test_forces, test_forces)}, Test MAE: {get_mae(predicted_test_forces, test_forces)}")

"""import cProfile
cProfile.runctx('model.train_epoch(data_loader, force_weight)', globals(), locals(), 'profile')

import pstats
stats = pstats.Stats('profile')
stats.strip_dirs().sort_stats('tottime').print_stats(100)

import os
os.remove('profile')"""

import cProfile

def run():
    for epoch in range(5):
        
        total_loss = model.train_epoch(train_data_loader, force_weight)

        predicted_train_energies, predicted_train_forces = model.predict_epoch(predict_train_data_loader)
        predicted_test_energies, predicted_test_forces = model.predict_epoch(predict_test_data_loader)

        print()
        print(f"Epoch number {epoch}, Total loss: {total_loss}")
        print(f"Energy errors: Train RMSE: {get_rmse(predicted_train_energies, train_energies)}, Train MAE: {get_mae(predicted_train_energies, train_energies)}, Test RMSE: {get_rmse(predicted_test_energies, test_energies)}, Test MAE: {get_mae(predicted_test_energies, test_energies)}")
        if do_forces:
            print(f"Force errors: Train RMSE: {get_rmse(predicted_train_forces, train_forces)}, Train MAE: {get_mae(predicted_train_forces, train_forces)}, Test RMSE: {get_rmse(predicted_test_forces, test_forces)}, Test MAE: {get_mae(predicted_test_forces, test_forces)}")

cProfile.runctx('run()', globals(), {'run': run}, 'profile')


import pstats
stats = pstats.Stats('profile')
stats.strip_dirs().sort_stats('tottime').print_stats(100)

import os
os.remove('profile')


