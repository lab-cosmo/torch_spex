import torch
import numpy as np

from equistore import TensorMap, Labels, TensorBlock

class PowerSpectrum(torch.nn.Module):

    def __init__(self, all_species):
        super(PowerSpectrum, self).__init__()

        self.all_species = all_species
        self.nu_plus_one_count = {}
        self.properties_values = {}
        self.selected_features = {}

    def forward(self, spex):

        l_max = 0
        for idx, block in spex.items():
            l_max = max(l_max, idx["lam"])

        keys = []
        blocks = []
        nu = 1

        properties_names = (
            [f"{name}" for name in spex.block(0).properties.names]
            + [f"{name[:-1]}{nu+1}" for name in spex.block(0).properties.names]
        )

        for a_i in self.all_species:

            if nu not in self.nu_plus_one_count:

                nu_plus_one_count = 0
                selected_features = {}
                properties_values = []

                for l in range(l_max+1):
                    selected_features[l] = []

                    block = spex.block(lam=l, a_i=a_i)

                    for q_nu in range(block.values.shape[-1]):
                        for q_1 in range(block.values.shape[-1]):

                            # if q_1 == q_nu: continue

                            properties_list = [[block.properties[name][q_nu] for name in block.properties.names] + [block.properties[name][q_1] for name in block.properties.names[:-1]] + [0]]
                            properties_values.append(properties_list)
                            selected_features[l].append([q_nu, q_1])
                            
                            nu_plus_one_count += 1

                keys_to_be_removed = []
                for key in selected_features.keys():
                    if len(selected_features[key]) == 0: 
                        keys_to_be_removed.append(key)  # No features were selected.
                    else:
                        selected_features[key] = torch.tensor(selected_features[key])

                for key in keys_to_be_removed:
                    selected_features.pop(key)

                self.nu_plus_one_count[nu] = nu_plus_one_count
                self.selected_features[nu] = selected_features
                self.properties_values[nu] = properties_values

            nu_plus_one_count = self.nu_plus_one_count[nu]
            selected_features = self.selected_features[nu]
            properties_values = self.properties_values[nu]

            block = spex.block(lam=0, a_i=a_i)
            data = torch.empty((len(block.samples), nu_plus_one_count), device=block.values.device)

            nu_plus_one_count = 0  # reset counter
            for l in range(l_max+1):  # l and lbda are now the same thing
                if l not in selected_features: continue  # No features are selected.

                cg = 1.0/np.sqrt(2*l+1)
                block = spex.block(lam=l, a_i=a_i)
                block = spex.block(lam=l, a_i=a_i)

                data[:, nu_plus_one_count:nu_plus_one_count+selected_features[l].shape[0]] = cg*torch.sum(block.values[:, :, selected_features[l][:, 0]]*block.values[:, :, selected_features[l][:, 1]], dim = 1, keepdim = False)
                from torch_spex.normalize import get_2_mom
                """
                print(l)
                print(torch.mean(block.values[:, :, selected_features[l][:, 0]]))
                print(torch.mean(block.values[:, :, selected_features[l][:, 1]]))
                print(torch.mean(block.values[:, :, selected_features[l][:, 0]]*block.values[:, :, selected_features[l][:, 1]]))
                print(torch.mean(data[:, nu_plus_one_count:nu_plus_one_count+selected_features[l].shape[0]]))
                print(get_2_mom(block.values[:, :, selected_features[l][:, 0]]))
                print(get_2_mom(block.values[:, :, selected_features[l][:, 1]]))
                print(get_2_mom(block.values[:, :, selected_features[l][:, 0]]*block.values[:, :, selected_features[l][:, 1]]))
                print(get_2_mom(data[:, nu_plus_one_count:nu_plus_one_count+selected_features[l].shape[0]]))
                """


                nu_plus_one_count += selected_features[l].shape[0]

            block = TensorBlock(
                values=data,
                samples=block.samples,
                components=[],
                properties=Labels(
                    names=properties_names,
                    values=np.asarray(np.vstack(properties_values), dtype=np.int32),
                ),
            )
            keys.append([a_i])
            blocks.append(block)

        LE_invariants = TensorMap(
            keys = Labels(
                names = ("a_i",),
                values = np.array(keys), # .reshape((-1, 2)),
            ), 
            blocks = blocks)

        return LE_invariants
