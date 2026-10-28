"""
Trainable bilateral filter

Author: Fabian Wagner
Contact: fabian.wagner@fau.de
"""

from typing import List

import torch
import bilateral_filter_nd_embedding_gpu_lib

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

class BilateralFilterFunctionNdEmbeddingGPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputTensor: torch.Tensor,
                sigma_t: torch.nn.Parameter, sigma_x: torch.nn.Parameter, sigma_y: torch.nn.Parameter, sigma_z: torch.nn.Parameter,
                embedding: torch.Tensor, hidden_width: int, hidden_layers: int, pad_t: int):
        
        assert inputTensor.shape[1] == 1, 'Currently only one channel images are supported'
        assert inputTensor.dim() == 6, 'Input shape must be [B, C, T, H, W, D]'

        inputTensor, embedding = inputTensor.contiguous(), embedding.contiguous()
        outputTensor, outputWeightsTensor = bilateral_filter_nd_embedding_gpu_lib.forward_gpu(inputTensor, float(sigma_t), float(sigma_x), float(sigma_y), float(sigma_z),
                                                                                              embedding, int(hidden_width), int(hidden_layers))
        ctx.save_for_backward(inputTensor, outputTensor, outputWeightsTensor, embedding)
        ctx.sigmas = tuple(float(s) for s in (sigma_t, sigma_x, sigma_y, sigma_z))
        ctx.hidden_width, ctx.hidden_layers, ctx.pad_t = int(hidden_width), int(hidden_layers), int(pad_t)

        return outputTensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):

        grad_output = grad_output.contiguous()
        inputTensor, outputTensor, outputWeightsTensor, embedding = ctx.saved_tensors
        hidden_width, hidden_layers, pad_t = ctx.hidden_width, ctx.hidden_layers, ctx.pad_t
        grad_input, dO_dsig_t, dO_dsig_x, dO_dsig_y, dO_dsig_z, grad_embedding = bilateral_filter_nd_embedding_gpu_lib.backward_gpu(grad_output, inputTensor, outputTensor, outputWeightsTensor, *ctx.sigmas, embedding, hidden_width, hidden_layers, pad_t)
        grad_sigmas = [torch.sum(grad_output * d_s)
                       for i, d_s in enumerate((dO_dsig_t, dO_dsig_x, dO_dsig_y, dO_dsig_z))]
        
        return (grad_input, grad_sigmas, grad_embedding, None, None, None)


class BilateralFilterNdEmbedding(torch.nn.Module):
    def __init__(self, axes=('t', 'x', 'y', 'z'),
                 sigma_t=0.0, sigma_x=0.0, sigma_y=0.0, sigma_z=0.0,
                 hidden_width=16, hidden_layers=2, 
                 pad_t=4, use_gpu=True):

        super().__init__()

        unknown = set(axes) - {'t', 'x', 'y', 'z'}
        if unknown or not axes:
            raise ValueError(f'axes must be non empty subset of t, x, y, z, got {axes}')


        init = {'t': sigma_t, 'x': sigma_x, 'y': sigma_y, 'z': sigma_z}
        self.axes = tuple(axes)
        for a in 'txyz':
            if a in self.axes:
                if float(init[a]) <= 0.0:
                    raise ValueError(f'active axis {a} needs sigma > 0, got {init[a]}')
                setattr(self, f'sigma_{a}', torch.nn.Parameter(torch.tensor(float(init[a]))))
            else:
                setattr(self, f'sigma_{a}', 0.0)

        self.hidden_width = int(hidden_width)
        self.hidden_layers = int(hidden_layers) - 1

        H = self.hidden_width
        L = self.hidden_layers
        self.range_net = torch.nn.Sequential(
            torch.nn.Linear(2, self.hidden_width),
            *[torch.nn.Linear(H, H) for _ in range(L)],
            torch.nn.Linear(H, 1),
        )

        with torch.no_grad():
            self.range_net[-1].weight.zero_()
            self.range_net[-1].bias.fill_(1.0)

        self.pad_t = pad_t if 't' in self.axes else 0
        self.use_gpu = use_gpu

    def reshaped_parameters(self):
        first, hidden, last = self.range_net[0], self.range_net[1:-1], self.range_net[-1]
        parts = [first.weight.reshape(-1), first.bias]
        if hidden:
            parts += [torch.stack([l.weight for l in hidden]).reshape(-1),
                      torch.stack([l.bias for l in hidden]).reshape(-1)]
        parts += [last.weight.reshape(-1), last.bias]
        return torch.cat(parts)


    def parameter_groups(self, lr):
        spatial = [getattr(self, f'sigma_{a}') for a in 'xyz' if isinstance(self, f'sigma_{a}', torch.nn.Parameter)]
        temporal = [self.sigma_t] if isinstance(self.sigma_t, torch.nn.Parameter) else []
        embedding = list(self.range_net.parameters())

        return [
            {'name': 'spatial', 'params': spatial, 'lr': lr['spatial'], 'weight_decay': 0.0},
            {'name': 'temporal', 'params': temporal, 'lr': lr['temporal'], 'weight_decay': 0.0},
            {'name': 'embedding', 'params': embedding, 'lr': lr['embedding'], 'weight_decay': 1e-5},
        ]

    def _get_sigmas(self):
        return (self.sigma_t, self.sigma_x, self.sigma_y, self.sigma_z)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 6, 'Input shape must be [B, C, T, D, H, W]'
        assert x.shape[1] == 1, 'Currently only one channel images are supported'

        x = x.permute(0, 1, 2, 4, 5, 3).contiguous()
        x = pad_circular_t(x, self.pad_t)
        embedding = self.reshaped_parameters().to(x.dtype)
        

        if self.use_gpu:
            output = BilateralFilterFunctionNdEmbeddingGPU.apply(x, *self._get_sigmas(), embedding, self.hidden_width, self.hidden_layers, self.pad_t)
        else:
            raise NotImplementedError('CPU is not supported')

        output = unpad_circular_t(output, self.pad_t)
        output = output.permute(0, 1, 2, 5, 3, 4).contiguous()
        return output

    def __repr__(self):
        s = ' '.join(f'sigma_{a}: {float(getattr(self, f"sigma_{a}")):.4f}' for a in 'txyz')
        numel = sum(p.numel() for p in self.range_net.parameters())
        return (f'BilateralFilterNdEmbedding({s} range net: 2 -> {self.hidden_width} x {self.hidden_layers + 1} -> 1, '
                f'{numel} weights')   
                