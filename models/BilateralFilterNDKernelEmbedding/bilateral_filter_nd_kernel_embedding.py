import math
from typing import Literal
import warnings

import torch

import bilateral_filter_nd_kernel_embedding_gpu_lib

def pad_circular_t(x: torch.Tensor, pad_t: int):
    if pad_t < 1:
        return x

    T = x.shape[2]
    if pad_t > T:
        idx = torch.arange(-pad_t, t + pad_t, device=x.device) % T
        return x.index_select(2, idx).contiguous()

    return torch.cat([x[:, :, -pad_t:], x, x[:, :, :pad_t]], dim=2).contiguous()

def unpad_circular(x: torch.Tensor, pad_t: int):
    if pad_t < 1:
        return x

    return x[:, :, pad_t:-pad_t]

def temporal_window(sigma_t: float, T: int):
    if sigma_t <= 0:
        return 1

    window = max(math.ceil(5.0 * sigma_t) | 1, 5)
    widest = T if T % 2 else T - 1
    return min(window, widest)

def gaussian_1d(sigma, window_size, dtype=torch.float64):
    d = torch.arange(0, window_size, dtype=dtype) - window_size // 2
    return torch.exp(-(d ** 2) / (2.0 * float(sigma) ** 2)).to(dtype)

def gaussian_4d(sigma_t, sigma_x, sigma_y, sigma_z, window_sizes, dtype=torch.float64):
    wt, wx, wy, wz, = window_sizes
    gt = gaussian_1d(sigma_t, wt, dtype)
    gx = gaussian_1d(sigma_x, wx, dtype)
    gy = gaussian_1d(sigma_y, wy, dtype)
    gz = gaussian_1d(sigma_z, wz, dtype)

    return (gt[:, None, None, None] * gx[None, :, None, None] * gy[None, None, :, None] * gz[None, None, None, :]).contiguous()

def default_window_sizes(sigma_t, sigma_x, sigma_y, sigma_z):
    def w(s):
        if s == 0.0:
            return 1
        return max(int(math.ceil(5.0 * float(s)) | 1), 5)

    return (w(sigma_t), w(sigma_x), w(sigma_y), w(sigma_z))

def inverse_softplus(y: torch.Tensor, eps: float = 1e-6):
    y = y.clamp_min(eps)
    return y + torch.log(-torch.expm1(-y))

def activation_to_key(act: Literal['linear', 'sigmoid', 'tanh', 'relu', 'leakyrelu']='tanh'):
    key = 0
    if act == 'linear':
        key = 0
    elif act == 'sigmoid':
        key = 1
    elif act == 'tanh':
        key = 2
    elif act == 'relu':
        key = 3
    elif act == 'leakyrelu':
        key = 4
    else:
        raise ValueError(f'Activation function must be one of (linear, sigmoid, tanh, relu, leakyrelu), go {act}')

    return key

class BilateralFilterFunctionNdKernelEmbeddingGPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputTensor: torch.Tensor, kernel: torch.Tensor, embedding: torch.Tensor,
                hidden_width: int, hidden_layers: int, activation: int, pad_t: int):
        
        assert inputTensor.shape[1] == 1, 'Currently only one channel images are supported'
        assert inputTensor.dim() == 6, 'Input shape must be [B, C, T, H, W, D]'

        inputTensor, kernel, embedding = inputTensor.contiguous(), kernel.contiguous(), embedding.contiguous()

        outputTensor, outputWeightsTensor = bilateral_filter_nd_kernel_embedding_gpu_lib.forward_gpu(inputTensor, kernel, embedding,
                                                                                                     int(hidden_width), int(hidden_layers), int(pad_t), int(activation))

        ctx.save_for_backward(inputTensor, outputTensor, outputWeightsTensor, kernel, embedding)
        ctx.hidden_width, ctx.hidden_layers, ctx.pad_t, ctx.act = int(hidden_width), int(hidden_layers), int (pad_t), int(activation)

        return outputTensor 

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        
        grad_output = grad_output.contiguous()
        inputTensor, outputTensor, outputWeightsTensor, kernel, embedding = ctx.saved_tensors
        hidden_width, hidden_layers, pad_t, act = ctx.hidden_width, ctx.hidden_layers, ctx.pad_t, ctx.act

        compute_input_grad = bool(ctx.needs_input_grad[0])
        grad_input, grad_kernel, grad_embedding = bilateral_filter_nd_kernel_embedding_gpu_lib.backward_gpu(grad_output, inputTensor, outputTensor, outputWeightsTensor, kernel, embedding, hidden_width, hidden_layers, pad_t, act, compute_input_grad)
        return (grad_input if compute_input_grad else None, grad_kernel, grad_embedding, None, None, None, None) 


class BilateralFilterNdKernelEmbedding(torch.nn.Module):
    def __init__(self,
                 sigma_t=0.0, sigma_x=0.0, sigma_y=0.0, sigma_z=0.0,
                 kernel_floor=1e-6,
                 hidden_width=16, hidden_layers=2, activation='tanh',
                 diff_scale=40.0, use_gpu=True, **kwargs):

        super().__init__()

        window_sizes = default_window_sizes(sigma_t, sigma_x, sigma_y, sigma_z)
        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.sigma_t = float(sigma_t)
        self.sigma_x = float(sigma_x)
        self.sigma_y = float(sigma_y)
        self.sigma_z = float(sigma_z)

        kernel = gaussian_4d(self.sigma_t, self.sigma_x, self.sigma_y, self.sigma_z, self.window_sizes)
        kernel = kernel / kernel.max()
        self.log_kernel = torch.nn.Parameter(torch.log(kernel + float(kernel_floor)))

        self.hidden_width = int(hidden_width)
        self.hidden_layers = int(hidden_layers - 1)
        self.activation = activation_to_key(activation)
        self.diff_scale = float(diff_scale)

        H = self.hidden_layers
        L = self.hidden_layers

        self.range_net = torch.nn.Sequential(
            torch.nn.Linear(4, H),
            *[torch.nn.Linear(H, H) for _ in self.hidden_layers],
            torch.nn.Linear(H, 1),
        )

        with torch.no_grad():
            self.range_net[-1].weight.zero_()
            self.range_net[-1].bias.fill_(1.0)

        self.use_gpu = use_gpu

        if kwargs:
            warnings.warn(f'Detected unknown initialization arguments: {kwargs}')

    def kernel_weights(self):
        k = torch.exp(self.log_kernel)
        return k / k.sum()

    def input_weights(self):
        scale = torch.ones(4, dtype=self.range_net[0].weight.dtype, device=self.range_net[0].weight.device)
        scale[2] = self.diff_scale
        return self.range_net[0].weight * scale

    def reshaped_parameters(self):
        first, hidden, last = self.range_net[0], self.range_net[1:-1], self.range_net[-1]
        parts = [self.input_weights(), first.bias]
        if hidden:
            parts += [torch.stack([l.weight for l in hidden]).reshape(-1),
                      torch.stack([l.bias for l in hidden]).reshape(-1)]
        parts += [last.weight.reshape(-1), last.bias]
        return torch.cat(parts).contiguous()

    def parameter_groups(self, lr):
        kernel = list(self.log_kernel)
        embedding = list(self.range_net.parameters())

        return [
            {'name': 'kernel', 'params': kernel, 'lr':lr['kernel'], 'weight_decay': 0.0},
            {'name': 'embedding', 'params': embedding, 'lr': lr['embedding'], 'weight_decay': 0.0},
        ]

    def forward(self, x: torch.Tensor):
        assert x.dim() == 6, 'Input must be of shape [B, C, T X, Y, Z]'
        assert x.shape[1] == 1, 'Currently only one channel images are supported'

        x = x.permute(0, 1, 2, 4, 5, 3).contiguous()
        pad_t = temporal_window(float(self.sigma_t), x.shape[2])
        x = pad_circular_t(x, pad_t)

        kernel = self.kernel_weights().to(dtype=x.dtype)
        embedding = self.reshaped_parameters().to(dtype=x.dtype)

        if self.use_gpu:
            output = BilateralFilterFunctionNdKernelEmbeddingGPU.apply(x, kernel, embedding, self.hidden_width, self.hidden_layers, self.activation, pad_t)
        else:
            raise NotImplementedError('Cpu is not supported')

        output = unpad_circular(output, pad_t)
        output = output.permute(0, 1, 2, 5, 3, 4).contiguous()
        return output

    def __repr__(self):
        numel_kernel = self.log_kernel.numel()
        numel_embedding = sum(p.numel() for p in self.range_net.parameters())
        return (f'BilateralFilterNdKernelEmbedding(kernel: {numel_kernel} weights,  '
                f'range net: 4 -> {self.hidden_width} x {self.hidden_layers + 1} -> 1,  '
                f'diff scale {self.diff_scale}, {numel_embedding} weights')




