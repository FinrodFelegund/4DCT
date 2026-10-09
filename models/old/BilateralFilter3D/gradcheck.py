"""
Calculation of gradcheck (PyTorch built-in function to check custom layer implementation)
for verifying a correct gradient implementation.
Documentation: https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html

Author: Fabian Wagner
Contact: fabian.wagner@fau.de
"""
from bilateral_filter_layer_3d import BilateralFilterFunction3dGPU
import torch
from torch.autograd import gradcheck

SHAPE = (2, 1, 10, 10, 10)
VALUES = {'sigma_x': 1.1, 'sigma_y': 1.1, 'sigma_z': 1.1, 'color_sigma': 0.5}

def sigmas(grad=None):
    return tuple(torch.tensor(v, dtype=torch.double, requires_grad=(k==grad)) for k, v in VALUES.items())

def constant_input(layer_bf):
    tensor_in = torch.full(SHAPE, 0.5, dtype=torch.double, device='cuda')
    out = layer_bf(tensor_in, *sigmas())
    print(bool(torch.isfinite(out).all()) and torch.allclose(out, tensor_in, atol=1e-6))

def gradient_input(layer_bf):
    tensor_in = torch.randn(SHAPE, dtype=torch.double, device='cuda', requires_grad=True)
    print(gradcheck(layer_bf, (tensor_in, *sigmas()), eps=1e-6, atol=1e-5))


def gradient_x(layer_bf):
    tensor_in = torch.randn(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(layer_bf, (tensor_in, *sigmas('sigma_x')), eps=1e-2, atol=1e-3))


def gradient_y(layer_bf):
    tensor_in = torch.randn(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(layer_bf, (tensor_in, *sigmas('sigma_y')), eps=1e-2, atol=1e-3))


def gradient_z(layer_bf):
    tensor_in = torch.randn(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(layer_bf, (tensor_in, *sigmas('sigma_z')), eps=1e-2, atol=1e-3))


def gradient_r(layer_bf):
    tensor_in = torch.randn(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(layer_bf, (tensor_in, *sigmas('color_sigma')), eps=1e-3, atol=1e-3))


layer_bf_gpu = BilateralFilterFunction3dGPU.apply


# Note that eps and atol are chosen according to the investigated parameter, as the
# filter parameters operate on different scales/regimes. For example, eps
# (step size for the numerical gradient) must be chosen fairly large for the spatial
# parameters in order to get meaningful gradients. The tolerance parameter atol
# is chosen according to the magnitude of the numerical gradients, determined by eps.

print('-------------------------------------------------------------')
print('Gradcheck is passed if no error occurs and \'True\' is printed.')
print('-------------------------------------------------------------')
print('Reproducing constant input:')
constant_input(layer_bf_gpu)

print('Gradient with respect to input:')
gradient_input(layer_bf_gpu)

print('Gradient with respect to sigma_x:')
gradient_x(layer_bf_gpu)

print('Gradient with respect to sigma_y:')
gradient_y(layer_bf_gpu)

print('Gradient with respect to sigma_z:')
gradient_z(layer_bf_gpu)

print('Gradient with respect to sigma_range:')
gradient_r(layer_bf_gpu)

