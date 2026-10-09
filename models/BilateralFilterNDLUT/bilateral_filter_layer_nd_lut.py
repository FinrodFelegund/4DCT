"""
Trainable bilateral filter layer.

Author: Fabian Wagner
Contact: fabian.wagner@fau.de
"""

import torch
import bilateral_filter_nd_lut_gpu_lib

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

def inverse_softplus(y: torch.Tensor, eps: float = 1e-6):
    y = y.clamp_min(eps)
    return y + torch.log(-torch.expm1(-y))



class BilaterFilterFunctionNdLUTGPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_img: torch.Tensor, sigma_t: torch.nn.Parameter, sigma_x: torch.nn.Parameter, sigma_y: torch.nn.Parameter, sigma_z: torch.nn.Parameter, lut: torch.Tensor):
        assert input_img.shape[1] == 1, 'Currently only one channel images are supported'
        assert input_img.dim() == 6, 'Input shape must be [B, C, T, H, W, D]'
        assert lut.dim() == 2 and lut.shape[0] == lut.shape[1], 'Look up table must be square tensor'

        input_img = input_img.contiguous()
        lut = lut.contiguous()
        outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_t, dO_dsig_x, dO_dsig_y, dO_dsig_z = bilateral_filter_nd_lut_gpu_lib.forward_gpu(input_img, float(sigma_t), float(sigma_x), float(sigma_y), float(sigma_z), lut)

        ctx.save_for_backward(input_img, outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_t, dO_dsig_x, dO_dsig_y, dO_dsig_z, lut)
        ctx.sigmas = tuple(float(s) for s in (sigma_t, sigma_x, sigma_y, sigma_z))

        return outputTensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):

        input_img, outputTensor, outputWeightsTensor, dO_dx_ki, d_dsig_t, d_sig_x, d_sig_y, d_sig_z, lut = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        grad_input, grad_lut = bilateral_filter_nd_lut_gpu_lib.backward_gpu(grad_output, input_img, outputTensor, outputWeightsTensor, dO_dx_ki, *ctx.sigmas, lut)
        grad_sigmas = [torch.sum(grad_output * d_s) if ctx.needs_input_grad[i + 1] else None
                       for i, d_s in enumerate((d_dsig_t, d_sig_x, d_sig_y, d_sig_z))]
        return (grad_input, *grad_sigmas, grad_lut)


class BilaterFilterNdLUT(torch.nn.Module):
    def __init__(self, color_sigma, axes=('t', 'x', 'y', 'z'),
                 sigma_t=0.0, sigma_x=0.0, sigma_y=0.0, sigma_z=0.0,
                 lut_bins=96, lut_max=1.0, symmetric_lut=True, use_gpu=True, pad_t=4):

        super().__init__()

        if not 0.0 < float(lut_max) <= 1.0:
            raise ValueError(f'lut_max must be in (0,1], got {lut_max}')

        unknow = set(axes) - ('t' , 'x', 'y', 'z')
        if unknow or not axes:
            raise ValueError(f'axes must be non-empty subset of t, x, y, z, got {axes}')

        init = {'t': sigma_t, 'x': sigma_x, 'y': sigma_y, 'z': sigma_z}
        self.axes = tuple(axes)
        for a in 'txyz':
            if a in self.axes:
                if float(init[a]) <= 0.0:
                    raise ValueError(f'active axis {a} needs sigma > 0, got {init[a]}')
                setattr(self, f'sigma_{a}', torch.nn.Parameter(torch.tensor(float(init[a]))))
            else:
                setattr(self, f'sigma_{a}', 0.0)

        self.pad_t = pad_t if 't' in self.axes else 0
        self.color_sigma_init = float(color_sigma)
        self.lut_bins, self.lut_max = int(lut_bins), float(lut_max)
        self.symmetric_lut = symmetric_lut
        self.use_gpu = use_gpu
        self.raw_lut = torch.nn.Parameter(inverse_softplus(gaussian_lut_2d(color_sigma / self.lut_max, self.lut_bins)))

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

    def parameter_groups(self, lr):
        spatial = [getattr(self, f'sigma_{a}') for a in 'xyz' if isinstance(getattr(self, f'sigma_{a}'), torch.nn.Parameter)]
        temporal = [self.sigma_t] if isinstance(self.sigma_t, torch.nn.Parameter) else []
        return [
            {'name': 'spatial', 'params': spatial, 'lr': lr['spatial'], 'weight_decay': 0.0},
            {'name': 'temporal', 'params': temporal, 'lr': lr['temporal'], 'weight_decay': 0.0},
            {'name': 'lut', 'params': [self.raw_lut], 'lr': lr['lut'], 'weight_decay': 0.0},
        ]

    def _get_sigmas(self):
        return self.sigma_t, self.sigma_x, self.sigma_y, self.sigma_z, self.color_sigma

    def forward(self, x: torch.Tensor):
        assert x.dim() == 6, 'Input must be shape [B, C, T, H, W, D]'
        assert x.shape[1] == 1, 'Currently only 1 channel immages are supported'

        x = x.permute(0, 1, 2, 4, 5, 3).contiguous() / self.lut_max
        x = pad_circular_t(x, self.pad_t)
        lut = self.activated_lut().to(x.dtype)

        if self.use_gpu:
            output = BilaterFilterFunctionNdLUTGPU.apply(x, self.sigma_t, self.sigma_x, self.sigma_y, self.sigma_z, lut)
        else:
            raise NotImplementedError('CPU is not supported')
        
        output = unpad_circular_t(output, self.pad_t)
        output = output.permute(0, 1, 2, 5, 3, 4).contiguous() * self.lut_max
        return output

    def __repr__(self):
        s = '  '.join(f'sigma_{a}: {float(getattr(self, f"sigma_{a}")):.4f}' for a in 'txyz')
        return (f'BilateralFilterNdLUT({s}  color_sigma (init): {self.color_sigma_init}  '
                f'lut: {self.lut_bins} bins up to {self.lut_max * 4024.0 - 1024.0:.0f} HU)')