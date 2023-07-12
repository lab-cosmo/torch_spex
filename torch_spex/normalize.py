import torch


class NormalizedActivationFunction(torch.nn.Module):

    def __init__(self, activation_function):
        super().__init__()
        with torch.no_grad():
            z = torch.randn(1000000, dtype=torch.get_default_dtype())
            self.normalization_factor = activation_function(z).pow(2).mean().pow(-0.5).item()
        self.activation_function = activation_function

    def forward(self, x):
        return self.normalization_factor * self.activation_function(x)


class NormalizedLinearLayerNoBias(torch.nn.Module):

    def __init__(self, linear_layer):
        super().__init__()
        linear_layer.weight.data.normal_(0.0, 1.0)
        n_in = linear_layer.weight.data.shape[0]
        self.normalization_factor = n_in ** (-0.5)
        self.linear_layer = linear_layer

    def forward(self, x):
        return self.normalization_factor * self.linear_layer(x)
    

class NormalizedLinearLayerWithBias(torch.nn.Module):

    def __init__(self, linear_layer):
        super().__init__()
        linear_layer.weight.data.normal_(0.0, 1.0)
        linear_layer.bias.data.zero_()
        n_in = linear_layer.weight.data.shape[0]
        self.normalization_factor = n_in ** (-0.5)
        self.linear_layer = linear_layer

    def forward(self, x):
        return self.normalization_factor * self.linear_layer(x)


def normalize_true(module):

    parameter_count = sum(1 for _ in module.parameters())

    if parameter_count == 0:  # Activation function
        normalized_module = NormalizedActivationFunction(module)

    elif parameter_count == 1:  # Linear layer without bias
        normalized_module = NormalizedLinearLayerNoBias(module)

    elif parameter_count == 2:  # Linear layer with bias
        normalized_module = NormalizedLinearLayerWithBias(module)

    else:
        raise ValueError("Normalization FAILED")

    return normalized_module


def normalize_false(module):
    return module
