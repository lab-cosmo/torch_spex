import torch
from .neighbor_list import get_neighbor_list


class NormalizedActivationFunction(torch.nn.Module):

    def __init__(self, activation_function):
        super().__init__()
        with torch.no_grad():
            z = torch.randn(1000000, dtype=torch.get_default_dtype())
            self.normalization_factor = activation_function(z).pow(2).mean().pow(-0.5).item()
        self.activation_function = activation_function

    def forward(self, x):
        return self.normalization_factor * self.activation_function(x)

class NormalizedEmbedding(torch.nn.Module):

    def __init__(self, linear_layer):
        super().__init__()
        linear_layer.weight.data.normal_(0.0, 1.0)
        self.linear_layer = linear_layer

    def forward(self, x):
        return self.linear_layer(x)

class NormalizedLinearLayerNoBias(torch.nn.Module):

    def __init__(self, linear_layer):
        super().__init__()
        linear_layer.weight.data.normal_(0.0, 1.0)
        n_in = linear_layer.weight.data.shape[1]
        self.normalization_factor = n_in ** (-0.5)
        self.linear_layer = linear_layer

    def forward(self, x):
        return self.normalization_factor * self.linear_layer(x)
    

class NormalizedLinearLayerWithBias(torch.nn.Module):

    def __init__(self, linear_layer):
        super().__init__()
        linear_layer.weight.data.normal_(0.0, 1.0)
        linear_layer.bias.data.zero_()
        n_in = linear_layer.weight.data.shape[1]
        self.normalization_factor = n_in ** (-0.5)
        self.linear_layer = linear_layer

    def forward(self, x):
        return self.normalization_factor * self.linear_layer(x)


def normalize_true(module_type, module):

    if module_type == "activation":  # Activation function
        normalized_module = NormalizedActivationFunction(module)

    elif module_type == "linear_no_bias":  # Linear layer without bias
        normalized_module = NormalizedLinearLayerNoBias(module)

    elif module_type == "linear_with_bias":  # Linear layer with bias
        normalized_module = NormalizedLinearLayerWithBias(module)

    elif module_type == "embedding":  # Linear layer with bias
        normalized_module = NormalizedEmbedding(module)

    else:
        raise ValueError("Normalization FAILED")

    return normalized_module


def normalize_false(module_type, module):
    return module


def get_average_number_of_neighbors(structures, r_cut):
    # Meant to be used on the training set
    n_total_centers = 0
    n_total_pairs = 0
    for structure in structures:
        centers, _, _ = get_neighbor_list(structure.positions, structure.pbc, structure.cell, r_cut)
        n_total_pairs += centers.shape[0]
        n_total_centers += structure.get_atomic_numbers().shape[0]
    # remember that the total number of pairs is double-counted. This is what we're after
    return n_total_pairs/n_total_centers


def get_2_mom(x):
    return torch.mean(x**2)

