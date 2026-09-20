"""
Gradcheck for the spatio-temporal bilateral filter with learnable center weight
(bilateral_filter_layer_4d_weightedcenter), GPU only.
Documentation: https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html
"""
from bilateral_filter_layer_4d_weightedcenter import BilateralFilterFunction4dsptGPUWeightedCenter
import torch
from torch.autograd import gradcheck

SHAPE = (1, 1, 6, 8, 8, 8)  # [B, C, T, X, Y, Z]
VALUES = {'sigma_x': 1.1, 'sigma_y': 1.1, 'sigma_z': 1.1, 'sigma_t': 1.1, 'color_sigma': 0.5, 'center_alpha': 0.7}


def params(grad=None):
    return tuple(torch.tensor(v, dtype=torch.double, requires_grad=(k == grad)) for k, v in VALUES.items())


def constant_input(layer_bf):
    # Holds for any center_alpha because the output is normalised by the weight sum.
    tensor_in = torch.full(SHAPE, 0.5, dtype=torch.double, device='cuda')
    out = layer_bf(tensor_in, *params())
    print(bool(torch.isfinite(out).all()) and torch.allclose(out, tensor_in, atol=1e-6))


def gradient_input(layer_bf):
    tensor_in = torch.randn(SHAPE, dtype=torch.double, device='cuda', requires_grad=True)
    print(gradcheck(layer_bf, (tensor_in, *params()), eps=1e-6, atol=1e-5))


def gradient_param(layer_bf, name, eps, atol):
    tensor_in = torch.randn(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(layer_bf, (tensor_in, *params(name)), eps=eps, atol=atol))


layer_bf_gpu = BilateralFilterFunction4dsptGPUWeightedCenter.apply

print('-------------------------------------------------------------')
print('Gradcheck is passed if no error occurs and \'True\' is printed.')
print('-------------------------------------------------------------')
print('Constant input is reproduced:')
constant_input(layer_bf_gpu)

print('Gradient with respect to input:')
gradient_input(layer_bf_gpu)

for sigma_name in ['sigma_x', 'sigma_y', 'sigma_z', 'sigma_t']:
    print(f'Gradient with respect to {sigma_name}:')
    gradient_param(layer_bf_gpu, sigma_name, eps=1e-2, atol=1e-3)

print('Gradient with respect to sigma_range:')
gradient_param(layer_bf_gpu, 'color_sigma', eps=1e-3, atol=1e-3)

print('Gradient with respect to center_alpha:')
gradient_param(layer_bf_gpu, 'center_alpha', eps=1e-3, atol=1e-3)
