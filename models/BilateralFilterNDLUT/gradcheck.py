"""
Gradcheck for the bilateral filter with learnable 4D kernel and learnable 2D
intensity lookup table (bilateral_filter_layer_4d_lut), GPU only.
Inputs are drawn in [0, 1] because the LUT is defined on that range.
The LUT is bilinear, so its derivative jumps at the bin knots. If only the
input check fails, re-run with another seed before suspecting the kernel.
Documentation: https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html
"""
from bilateral_filter_layer_nd_lut import BilaterFilterFunctionNdLUTGPU
import torch
from torch.autograd import gradcheck

SHAPE = (1, 1, 6, 5, 9, 11)  # [B, C, T, X, Y, Z]

BINS = 8
NONDET_TOL = 1e-6           # kernel and LUT gradients use atomicAdd
VALUES = {'sigma_t': 1.1, 'sigma_x': 1.1, 'sigma_y': 1.1, 'sigma_z': 1.1}

def sigmas(grad=None):
    return tuple(torch.tensor(v, dtype=torch.double, requires_grad=(k==grad)) for k, v in VALUES.items())

torch.manual_seed(0)


def lut(requires_grad=False, sigma_r=0.3):
    grid = torch.linspace(0.0, 1.0, BINS, dtype=torch.double)
    table = torch.exp(-((grid[:, None] - grid[None, :]) ** 2) / (2.0 * sigma_r ** 2))
    return table.contiguous().cuda().requires_grad_(requires_grad)


def constant_input(layer_bf):
    tensor_in = torch.full(SHAPE, 0.5, dtype=torch.double, device='cuda')
    out = layer_bf(tensor_in,  *sigmas(), lut())
    print(bool(torch.isfinite(out).all()) and torch.allclose(out, tensor_in, atol=1e-6))


def gradient_input(layer_bf):
    tensor_in = torch.rand(SHAPE, dtype=torch.double, device='cuda', requires_grad=True)
    print(gradcheck(layer_bf, (tensor_in,  *sigmas(), lut()), eps=1e-6, atol=1e-5, nondet_tol=NONDET_TOL))

def gradient_t(layer_bf):
    tensor_in = torch.randn(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(layer_bf, (tensor_in, *sigmas('sigma_t'), lut()), eps=1e-2, atol=1e-3))

def gradient_x(layer_bf):
    tensor_in = torch.randn(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(layer_bf, (tensor_in, *sigmas('sigma_x'), lut()), eps=1e-2, atol=1e-3))

def gradient_y(layer_bf):
    tensor_in = torch.randn(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(layer_bf, (tensor_in, *sigmas('sigma_y'), lut()), eps=1e-2, atol=1e-3))

def gradient_z(layer_bf):
    tensor_in = torch.randn(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(layer_bf, (tensor_in, *sigmas('sigma_z'), lut()), eps=1e-2, atol=1e-3))

def gradient_lut(layer_bf):
    tensor_in = torch.rand(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(layer_bf, (tensor_in,  *sigmas(), lut(requires_grad=True)), eps=1e-6, atol=1e-5, nondet_tol=NONDET_TOL))


layer_bf_gpu = BilaterFilterFunctionNdLUTGPU.apply

print('-------------------------------------------------------------')
print('Gradcheck is passed if no error occurs and \'True\' is printed.')
print('-------------------------------------------------------------')
print('Constant input is reproduced:')
constant_input(layer_bf_gpu)

print('Gradient with respect to input:')
gradient_input(layer_bf_gpu)

print('Gradient with respect to sigma t:')
gradient_t(layer_bf_gpu)

print('Gradient with respect to sigma x:')
gradient_x(layer_bf_gpu)

print('Gradient with respect to sigma y:')
gradient_y(layer_bf_gpu)

print('Gradient with respect to sigma z:')
gradient_z(layer_bf_gpu)

print('Gradient with respect to lut:')
gradient_lut(layer_bf_gpu)
