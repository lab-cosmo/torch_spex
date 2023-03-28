import torch
import copy


def check_forces_by_finite_differences(model, structure):

    delta = 1e-6

    _, backward_forces = model([structure], is_training=False)

    structure_forces = []
    for atom_index in range(len(structure.positions)):
        atom_forces = []
        for position_index in range(3):
            structure_plus = copy.deepcopy(structure)
            structure_minus = copy.deepcopy(structure)
            structure_plus.positions[atom_index, position_index] += delta
            structure_minus.positions[atom_index, position_index] -= delta
            energy_pair, _ = model([structure_plus, structure_minus])
            force = -(energy_pair[0] - energy_pair[1])/(2*delta)
            atom_forces.append(force)
        atom_forces = torch.tensor(atom_forces)
        structure_forces.append(atom_forces)
    finite_difference_forces = torch.stack(structure_forces, dim=0)

    assert torch.allclose(backward_forces, finite_difference_forces)
    print("Finite differences check passed successfully!")


if __name__ == "__main__":
    pass


