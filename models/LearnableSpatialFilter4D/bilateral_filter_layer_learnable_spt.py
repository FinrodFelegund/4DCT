"""
Trainable bilateral filter layer.

Author: Fabian Wagner
Contact: fabian.wagner@fau.de
"""
import torch
import torch.nn as nn
import bilateralfilter4dlearnablespt_gpu_lib

import math


def pad_circular_t(x: torch.Tensor, pad_t: int = 4):
    T = x.shape[2]
    if pad_t < 1:
        return x
    
    if pad_t > T:
        idx = torch.arange(-pad_t, T + pad_t, device=x.device) % T
        return x.index_select(2, idx).contiguous()
    return torch.cat([x[:, :, T - pad_t:], x, x[:, :, :pad_t]], dim=2).contiguous()

def unpad_circular_t(x: torch.Tensor, pad_t: int = 4):
    if pad_t < 1:
        return x
    return x[:, :, pad_t:-pad_t]

def gaussian_1d(sigma, window_size, dtype=torch.float32):
    d = torch.arange(window_size, dtype=torch.float64) - (window_size // 2)
    d = torch.exp(-(d ** 2) / (2.0 * float(sigma) ** 2)).to(dtype=dtype)
    return d

def gaussian_4d(sigma_t, sigma_x, sigma_y, sigma_z, window_sizes, dtype=torch.float32):

    wt, wx, wy, wz = window_sizes
    gt = gaussian_1d(sigma_t, wt, dtype)
    gx = gaussian_1d(sigma_x, wx, dtype)
    gy = gaussian_1d(sigma_y, wy, dtype)
    gz = gaussian_1d(sigma_z, wz, dtype)
    kernel = (gt[:, None, None, None] *
              gx[None, :, None, None] *
              gy[None, None, :, None] *
              gz[None, None, None, :])
    return kernel.contiguous()

def default_window_sizes(sigma_t, sigma_x, sigma_y, sigma_z, window_size_t=9):
    def w(s):
        return max(int(math.ceil(5.0 * float(s))) | 1, 5)

    assert window_size_t % 2 == 1, 'window_size_t must be odd'
    return (window_size_t, w(sigma_x), w(sigma_y), w(sigma_z))

class BilateralFilterFunction4dlearnablesptGPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_img: torch.Tensor, kernel: torch.Tensor, color_sigma):
        assert input_img.dim() == 6, 'Input shape must be [B, C, T, H, W, D]'
        assert input_img.shape[1] == 1, 'Currently only 1 channel scans are supported'
        assert kernel.dim() == 4, 'Kernel shape must be [T, H, W, D]'

        input_img = input_img.contiguous()
        kernel = kernel.contiguous()

        outputTensor, outputWeightsTensor, d0_dx_ki, d0_dsig_r = bilateralfilter4dlearnablespt_gpu_lib.forward_4dlearnablespt_gpu(input_img, kernel, color_sigma)

        ctx.save_for_backward(
            input_img,
            kernel,
            color_sigma,
            outputTensor,
            outputWeightsTensor,
            d0_dx_ki,
            d0_dsig_r,
        )

        return outputTensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.contiguous()
        input_img, kernel, color_sigma, outputTensor, outputWeightsTensor, d0_dx_ki, d0_dsig_r = ctx.saved_tensors

        grad_input, grad_kernel = bilateralfilter4dlearnablespt_gpu_lib.backward_4dlearnablespt_gpu(grad_output,
                                                                                                    input_img,
                                                                                                    outputTensor,
                                                                                                    outputWeightsTensor,
                                                                                                    d0_dx_ki,
                                                                                                    kernel,
                                                                                                    color_sigma)
        grad_color_sigma = torch.sum(grad_output * d0_dsig_r)
        return grad_input, grad_kernel, grad_color_sigma


class BilateralFilter4dlearnablespt(nn.Module):
    def __init__(self, sigma_x, sigma_y, sigma_z, sigma_t, color_sigma, 
                 window_sizes=None,
                 window_size_t=9,
                 pad_t=4,
                 use_gpu=True):

        super().__init__()

        self.use_gpu = use_gpu

        if window_sizes == None:
            window_sizes = default_window_sizes(sigma_t, sigma_x, sigma_y, sigma_z, window_size_t=window_size_t)

        for w in window_sizes:
            assert w % 2 == 1, 'All window sizes must be odd'

        self.window_sizes = tuple(int(w) for w in window_sizes)

        kernel = gaussian_4d(sigma_t, sigma_x, sigma_y, sigma_z, self.window_sizes)
        self.kernel = nn.Parameter(kernel)

        self.color_sigma = nn.Parameter(torch.tensor(float(color_sigma)), requires_grad=True)
        self.pad_t = self.window_sizes[0] // 2

    def parameter_groups(self, lr_kernel=1e-3, lr_sigma=1e-3):
        groups = [
            {
                'params': [self.kernel],
                'lr': lr_kernel,
                'weight_decay': 0.0,
            }, 
            {
                'params': [self.color_sigma],
                'lr': lr_sigma,
                'weight_decay': 0.0,
            }
        ]
        return groups



    def forward(self, input_tensor: torch.Tensor):

        assert len(input_tensor.shape) == 6, 'Input shape of 3d bilateral filter layer must equal [B, C, T, X, Y, Z].'
        assert input_tensor.shape[1] == 1, 'Currently channel dimensions >1 are not supported.'

        # Choose between CPU processing and CUDA acceleration.
        input_tensor = input_tensor.permute(0, 1, 2, 4, 5, 3).contiguous()
        input_tensor = pad_circular_t(input_tensor, self.pad_t)
        kernel = self.kernel.to(dtype=input_tensor.dtype)
        color_sigma = self.color_sigma.to(dtype=input_tensor.dtype)

        if self.use_gpu:
            output = BilateralFilterFunction4dlearnablesptGPU.apply(input_tensor,
                                                                    kernel,
                                                                    color_sigma)
        else:
            raise RuntimeError('CPU is not supported')

        output = unpad_circular_t(output, self.pad_t)

        output = output.permute(0, 1, 2, 5, 3, 4).contiguous()
        return output
