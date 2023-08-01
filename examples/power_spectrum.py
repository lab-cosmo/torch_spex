import torch
import numpy as np

from equistore import TensorMap, Labels, TensorBlock

class PowerSpectrum(torch.nn.Module):

    def __init__(self, l_max, all_species):
        super(PowerSpectrum, self).__init__()

        self.l_max = l_max
        self.all_species = all_species
        

    def forward(self, spex):

        keys = []
        blocks = []
        for a_i in self.all_species:
            ps_values_ai = []
            for l in range(self.l_max+1):
                cg = 1.0/np.sqrt(2*l+1)
                block_ai_l = spex.block(lam=l, a_i=a_i)
                c_ai_l = block_ai_l.values

                # same as this:
                # ps_ai_l = cg*torch.einsum("ima, imb -> iab", c_ai_l, c_ai_l)
                # but faster: 
                ps_ai_l = cg*torch.sum(c_ai_l.unsqueeze(2)*c_ai_l.unsqueeze(3), dim=1)

                ps_ai_l = ps_ai_l.reshape(c_ai_l.shape[0], c_ai_l.shape[2]**2)
                ps_values_ai.append(ps_ai_l)
            ps_values_ai = torch.concatenate(ps_values_ai, dim=-1)

            block = TensorBlock(
                values=ps_values_ai,
                samples=block_ai_l.samples,
                components=[],
                properties=Labels.range("property", ps_values_ai.shape[-1])
            )
            keys.append([a_i])
            blocks.append(block)

        power_spectrum = TensorMap(
            keys = Labels(
                names = ("a_i",),
                values = np.array(keys), # .reshape((-1, 2)),
            ), 
            blocks = blocks
        )

        return power_spectrum
