import torch
from .le import get_le_spliner


class RadialBasis(torch.nn.Module):

    def __init__(self, hypers, device) -> None:
        super().__init__()

        self.n_max_l, self.spliner = get_le_spliner(hypers["E_max"], hypers["r_cut"], device=device)
        self.l_max = len(self.n_max_l) - 1

    def forward(self, r):

        radial_functions = self.spliner.compute(r)

        radial_basis = []
        index = 0
        for l in range(self.l_max+1):
            radial_basis.append(
                radial_functions[:, index:index+self.n_max_l[l]]
            )
            index += self.n_max_l[l]

        return radial_basis


class SphericalBesselFirstKind(torch.autograd.Function):

    @staticmethod
    def forward(ctx, l, x):

        assert(len(x.shape) == 1)
        output = spherical_bessel.first_kind_forward(l, x)
        ctx.l = l
        ctx.save_for_backward(x)
        return output

    @staticmethod
    def backward(ctx, d_loss_d_output):

        l = ctx.l
        x, = ctx.saved_tensors
        d_output_d_x = spherical_bessel.first_kind_backward(l, x)

        return None, d_loss_d_output * d_output_d_x


class SphericalBesselSecondKind(torch.autograd.Function):

    @staticmethod
    def forward(ctx, l, x):

        output = spherical_bessel.second_kind_forward(l, x)
        ctx.l = l
        ctx.save_for_backward(x)
        return output

    @staticmethod
    def backward(ctx, d_loss_d_output):

        l = ctx.l
        x, = ctx.saved_tensors
        d_output_d_x = spherical_bessel.second_kind_backward(l, x)

        return None, d_loss_d_output * d_output_d_x
