## Spherical expansion coeffiecients for atom-centered densities

This is a small library that computes spherical expansions in Pytorch.
Compared to other implementations, it offers backpropagation and GPU support.

### Installation

```bash
pip install .
```

#### Enforcing CPU installation

We determine if a NVIDIA card is present using `nvidia-smi` and decide based on this, if the CPU or CUDA version of pytorch is installed.
To always install the CPU version, please use:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu .
```

### Examples

The examples folder contains a BPNN built using this spherical 
expansion code. This NN achieves decent accuracy on methane and rMD17.
