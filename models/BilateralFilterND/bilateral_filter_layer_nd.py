"""
Trainable bilateral filter layer Nd.

Author: Fabian Wagner
Contact: fabian.wagner@fau.de
"""

import torch
import bilateral_filter_layer_nd_gpu_lib

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

class BilateralFilterFunctionNdGPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_img, sigma_t, sigma_x, sigma_y, sigma_z, color_sigma):
        assert input_img.shape[1] == 1, 'Currently only one channel images are supported'
        assert input_img.dim() == 6, 'Input shape must be [B, C, T, H, W, D]'

        input_img = input_img.contiguous()
        outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_t, dO_dsig_x, dO_dsig_y, dO_dsig_z, dO_dsig_color = bilateral_filter_layer_nd_gpu_lib.forward_gpu(input_img, float(sigma_t), float(sigma_x), float(sigma_y), float(sigma_z), float(color_sigma))

        ctx.save_for_backward(input_img,
                              outputTensor,
                              outputWeightsTensor,
                              dO_dx_ki,
                              dO_dsig_t,
                              dO_dsig_x,
                              dO_dsig_y,
                              dO_dsig_z,
                              dO_dsig_color)
        ctx.sigmas = tuple(float(s) for s in (sigma_t, sigma_x, sigma_y, sigma_z, color_sigma))

        return outputTensor

    @staticmethod
    def backward(ctx, grad_output):

        grad_output = grad_output.contiguous()
        input_img, outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_t, dO_dsig_x, dO_dsig_y, dO_dsig_z, dO_dsig_color = ctx.saved_tensors

        grad_input = bilateral_filter_layer_nd_gpu_lib.backward_gpu(grad_output, input_img, outputTensor, outputWeightsTensor, dO_dx_ki, *ctx.sigmas)
        grad_sigmas = [torch.sum(grad_output * d_s) if ctx.needs_input_grad[i + 1] else None
                       for i, d_s in enumerate((dO_dsig_t, dO_dsig_x, dO_dsig_y, dO_dsig_z, dO_dsig_color))]

        return (grad_input, *grad_sigmas)


class BilateralFilterNd(torch.nn.Module):
    def __init__(self, color_sigma, axes=('t', 'x', 'y', 'z'),
                 sigma_t=0.0, sigma_x=0.0, sigma_y=0.0, sigma_z=0.0,
                 pad_t=4, use_gpu=True):

        super().__init__()
        self.use_gpu = use_gpu

        unknown = set(axes) - {'t', 'x', 'y', 'z'}
        if unknown or not axes:
            raise ValueError(f'axes must be non empty subset of t, x, y, z, got {axes}')

        if float(color_sigma) <= 0.0:
            raise ValueError(f'color sigma needs sigma > 0, got {color_sigma}')

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
        self.color_sigma = torch.nn.Parameter(torch.tensor(float(color_sigma)))

    def parameter_groups(self, lr):
        spatial = [getattr(self, f'sigma_{a}') for a in 'xyz' if isinstance(getattr(self, f'sigma_{a}'), torch.nn.Parameter)]
        temporal = [self.sigma_t] if isinstance(self.sigma_t, torch.nn.Parameter) else []
        rng = [self.color_sigma]

        return [
            {'name': 'spatial', 'params': spatial, 'lr': lr['spatial'], 'weight_decay': 0.0},
            {'name': 'temporal', 'params': temporal, 'lr': lr['temporal'], 'weight_decay': 0.0},
            {'name': 'range', 'params': rng, 'lr': lr['range'], 'weight_decay': 0.0},
        ]

    def _get_sigmas(self):
        return self.sigma_t, self.sigma_x, self.sigma_y, self.sigma_z, self.color_sigma

    def forward(self, x: torch.Tensor):
        assert x.dim() == 6, 'Input must be shape [B, C, T, H, W, D]'
        assert x.shape[1] == 1, 'Currently only 1 channel images are supported'

        x = x.permute(0, 1, 2, 4, 5, 3).contiguous()
        x = pad_circular_t(x, self.pad_t)

        if self.use_gpu:
            output = BilateralFilterFunctionNdGPU.apply(x, self.sigma_t, self.sigma_x, self.sigma_y, self.sigma_z, self.color_sigma)
        else:
            raise NotImplementedError('CPU is not supported')

        output = unpad_circular_t(output, self.pad_t)
        return output.permute(0, 1, 2, 5, 3, 4).contiguous()

    def __repr__(self):
        s = ' '.join(f'sigma_{a}: {float(getattr(self, f"sigma_{a}")):.4f}' for a in 'txyz')
        s = s + f' color_sigma: {float(self.color_sigma.item())}'
        return (f'BilateralFilterNd({s})')


        