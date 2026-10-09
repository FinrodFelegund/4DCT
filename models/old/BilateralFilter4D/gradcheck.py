"""
Gradcheck for the temporal bilateral filter layer (bilateral_filter_layer_4d), GPU only.
The autograd.Function works on [N, C, T]; every voxel of the volume is one row.
Documentation: https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html
"""
from bilateral_filter_layer_4d import BilateralFilterFunction4dGPU
import torch
from torch.autograd import gradcheck

SHAPE = (8, 1, 8)  # [N, C, T]
VALUES = {'sigma_t': 1.1, 'color_sigma': 0.5}


def sigmas(grad=None):
    return tuple(torch.tensor(v, dtype=torch.double, requires_grad=(k == grad)) for k, v in VALUES.items())


def constant_input(layer_bf):
    tensor_in = torch.full(SHAPE, 0.5, dtype=torch.double, device='cuda')
    out = layer_bf(tensor_in, *sigmas())
    print(bool(torch.isfinite(out).all()) and torch.allclose(out, tensor_in, atol=1e-6))


def gradient_input(layer_bf):
    tensor_in = torch.randn(SHAPE, dtype=torch.double, device='cuda', requires_grad=True)
    print(gradcheck(layer_bf, (tensor_in, *sigmas()), eps=1e-6, atol=1e-5))


def gradient_t(layer_bf):
    tensor_in = torch.randn(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(layer_bf, (tensor_in, *sigmas('sigma_t')), eps=1e-2, atol=1e-3))


def gradient_r(layer_bf):
    tensor_in = torch.randn(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(layer_bf, (tensor_in, *sigmas('color_sigma')), eps=1e-3, atol=1e-3))


layer_bf_gpu = BilateralFilterFunction4dGPU.apply

print('-------------------------------------------------------------')
print('Gradcheck is passed if no error occurs and \'True\' is printed.')
print('-------------------------------------------------------------')
print('Constant input is reproduced:')
constant_input(layer_bf_gpu)

print('Gradient with respect to input:')
gradient_input(layer_bf_gpu)

print('Gradient with respect to sigma_t:')
gradient_t(layer_bf_gpu)

print('Gradient with respect to sigma_range:')
gradient_r(layer_bf_gpu)
