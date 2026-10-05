"""
Trainable bilateral filter layer.

Author: Fabian Wagner
Contact: fabian.wagner@fau.de
"""

import bilateralfilterndlut_gpu_lib
import torch

def pad_circular_t(x: torch.Tensor, pad_t: int):
    if pad_t < 1:
        return x

    T = x.shape[2]
    if pad_t > T:
        idx = torch.arange(-pad_t, T + pad_t, device=x.device) % T
        return x.index_select(2, idx).contiguous()
    return torch.cat([x[:, :, -pad_t:], x, x[:, :, :pad_t]], dim=2).contiguous()

def unpad_circular_t(x: torch.Tensor, pad_t: int):
    if pad_t < 1:
        return x
    return x[:, :, pad_t:-pad_t]

def gaussian_lut_2d(color_sigma: float, bins: int, dtype=torch.float64):
    """F[p, q] = exp(-(a_p - a_q)^2 / (2 sigma_r)), a_p = p / (bins - 1)
    Reproduces the gaussian range kernel initially.
    """

    a = torch.linspace(0.0, 1.0, bins, dtype=torch.float64)
    d = a[:, None] - a[None, :]
    return torch.exp(-(d ** 2) / (2.0 * float(color_sigma) ** 2)).contiguous()

def inverse_softplus(y: torch.Tensor, eps: float = 1e-12):
    y = y.clamp_min(eps)
    return y + torch.log(-torch.expm1(-y))



class BilaterFilterFunctionNdLUTGPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_img: torch.Tensor, sigma_x: torch.nn.Parameter, sigma_y: torch.nn.Parameter, sigma_z: torch.nn.Parameter, sigma_t: torch.nn.Parameter, lut: torch.Tensor):
        assert input_img.shape[1] == 1, 'Currently only one channel images are supported'
        assert input_img.dim() == 6, 'Input shape must be [B, C, T, H, W, D]'
        assert lut.dim() == 2 and lut.shape[0] == lut.shape[1], 'Look up table must be square tensor'

        input_img = input_img.contiguous()
        lut = lut.contiguous()
        outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_x, dO_dsig_y, dO_dsig_z, dO_dsig_t = bilateralfilterndlut_gpu_lib.forward_gpu(input_img, sigma_x, sigma_y, sigma_z, sigma_t, lut)

        ctx.save_for_backward(input_img, sigma_x, sigma_y, sigma_z, sigma_t, lut, outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_x, dO_dsig_y, dO_dsig_z, dO_dsig_t)

        return outputTensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.contiguous()
        grad_sig_x = None
        grad_sig_y = None
        grad_sig_z = None
        grad_sig_t = None

        input_img = ctx.saved_tensors[0]
        sigma_x = ctx.saved_tensors[1]
        sigma_y = ctx.saved_tensors[2]
        sigma_z = ctx.saved_tensors[3]
        sigma_t = ctx.saved_tensors[4]
        lut = ctx.saved_tensors[5]
        outputTensor = ctx.saved_tensors[6]
        outputWeightsTensor = ctx.saved_tensors[7]
        dO_dx_ki = ctx.saved_tensors[8]
        dO_dsig_x = ctx.saved_tensors[9]
        dO_dsig_y = ctx.saved_tensors[10]
        dO_dsig_z = ctx.saved_tensors[11]
        dO_dsig_t = ctx.saved_tensors[12]

        grad_sig_x = torch.sum(grad_output * dO_dsig_x) if sigma_x else 0
        grad_sig_y = torch.sum(grad_output * dO_dsig_y) if sigma_y else 0
        grad_sig_z = torch.sum(grad_output * dO_dsig_z) if sigma_z else 0
        grad_sig_t = torch.sum(grad_output * dO_dsig_t) if sigma_t else 0

        grad_output_tensor = bilateralfilterndlut_gpu_lib.backward_gpu(grad_output,
                                                                    input_img,
                                                                    outputTensor,
                                                                    outputWeightsTensor,
                                                                    dO_dx_ki,
                                                                    sigma_x,
                                                                    sigma_y,
                                                                    sigma_z,
                                                                    sigma_t,
                                                                    lut)

        return grad_output_tensor, grad_sig_x, grad_sig_y, grad_sig_z, grad_sig_t, lut

    class BilaterFilterNdLUT(torch.nn.Module):
        def __init__(self,
                     sigma_x, sigma_y, sigma_z, sigma_t, color_sigma,
                     lut_bins=32,
                     lut_max=1.0,
                     symmetric_lut=True,
                     pad_t=4,
                     use_gpu=True):

            super().__init__()

            self.sigma_x = torch.nn.Parameter(torch.tensor(float(sigma_x)))
            self.sigma_y = torch.nn.Parameter(torch.tensor(float(sigma_y)))
            self.sigma_z = torch.nn.Parameter(torch.tensor(float(sigma_z)))
            self.sigma_t = torch.nn.Parameter(torch.tensor(float(sigma_t)))

            self.color_sigma = float(color_sigma)

            self.lut_bins = int(lut_bins)
            self.lut_max = float(lut_max)
            self.symmetric_lut = symmetric_lut
            self.pad_t = pad_t
            self.use_gpu = use_gpu

            if not 0.0 < lut_max <= 1.0:
                raise ValueError(f'lut_max must lie in (0, 1], got {lut_max}')

            lut0 = gaussian_lut_2d(self.color_sigma / self.lut_max, self.lut_bins)
            self.raw_lut = torch.nn.Parameter(inverse_softplus(lut0))

        def activated_lut(self):
            lut = torch.nn.functional.softplus(self.raw_lut)
            if self.symmetric_lut:
                lut = (lut + lut.t()) / 2

            return lut / lut.diagonal().mean()

        def lut_smoothness_penalty(self):
            lut = self.activated_lut()
            d_row = lut[2:, :] - 2 * lut[1:-1, :] + lut[:-2, :]
            d_col = lut[:, 2:] - 2* lut[:, 1:-1] + lut[:, :-2]

            return (d_row ** 2).mean() + (d_col ** 2).mean()

        def parameter_groups(self, lr_sigmas=1e-3, lr_lut=1e-3):
            return [
                {'params': [self.sigma_x, self.sigma_y, self.sigma_z, self.sigma_t], 'lr': lr_sigmas, 'weight_decay': 0.0},
                {'params': [self.raw_lut], 'lr': lr_lut, 'weight_decay': 0.0},
            ]

        def _get_sigmas(self):
            return self.sigma_x, self.sigma_y, self.sigma_z, self.color_sigma

        def forward(self, x: torch.Tensor):
            assert x.dim() == 6, 'Input must be shape [B, C, T, H, W, D]'
            assert x.shape[1] == 1, 'Currently only 1 channel immages are supported'

            x = x.permute(0, 1, 2, 4, 5, 3).contiguous()
            if self.sigma_t:
                x = pad_circular_t(x, self.pad_t)
            lut = self.activated_lut().to(x.dtype)

            if self.use_gpu:
                output = BilaterFilterFunctionNdLUTGPU.apply(x, self.sigma_x, self.sigma_y, self.sigma_z, self.sigma_t, lut)
            else:
                raise NotImplementedError('CPU is not supported')
            
            if self.sigma_t:
                output = unpad_circular_t(output, self.pad_t)
            output = output.permute(0, 1, 2, 5, 3, 4).contiguous()

        def __repr__(self):
            return (f'BilaterFilterNdLUT(sigma_t: {self.sigma_t.item()} sigma_x: {self.sigma_x.item})  '
                    f'sigma_y: {self.sigma_y.item()} sigma_z: {self.sigma_z.item()} color_sigma (init): {self.color_sigma}  '
                    f'lut: {self.lut_bins} bins up to {self.lut_max * 4024.0 - 1024.0:.0f} HU')