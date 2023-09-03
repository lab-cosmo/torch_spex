import torch
from metatensor.torch import Labels


def one_hot(labels: Labels, dimension: Labels) -> torch.Tensor:

    if len(dimension.names) != 1:
        raise ValueError(
            "only one label dimension can be extracted as one-hot "
            "encoding. The `dimension` labels contains "
            f"{len(dimension.names)} names"
        )

    name = dimension.names[0]
    possible_labels = dimension.column(name)

    original_labels = labels.column(name)  # assuming the name is in the labels

    indices = torch.where(
        original_labels.reshape(original_labels.shape[0], 1) == possible_labels
    )[1]
    assert indices.shape[0] == labels.values.shape[0]  # if not, it means that some values not present in the dimension were found in the labels
    
    one_hot_array = torch.eye(possible_labels.shape[0])[indices]
    return one_hot_array
