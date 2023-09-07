import torch
from typing import List, Optional

def compute_forces(
    energy: torch.Tensor, positions: torch.Tensor, is_training: bool = True
) -> torch.Tensor:
    grad_outputs : Optional[List[Optional[torch.Tensor]]] = [torch.ones_like(energy)]
    gradient = torch.autograd.grad(
        outputs=[energy],
        inputs=[positions],
        grad_outputs=grad_outputs,
        retain_graph=is_training,
        create_graph=is_training,
    )[0]
    if gradient is None:
raise ValueError(
  "Unexpected None value for computed gradient. torch.autograd.grad input arguments:\n"
  f"  outputs={[energy]!r},\n"
  f"  inputs={[positions]!r},\n"
  f"  retain_graph={is_training},\n"
  f"  create_graph={is_training}")
    else:
        return -gradient


if __name__ == "__main__":
    
    positions = torch.tensor(([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), requires_grad=True)
    energies = torch.sum(positions, dim=0)

    forces = compute_forces(energies, positions)
    print(forces)
