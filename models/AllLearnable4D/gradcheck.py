"""
Gradcheck for the bilateral filter with learnable 4D kernel and learnable 2D
intensity lookup table (bilateral_filter_layer_4d_lut), GPU only.
Inputs are drawn in [0, 1] because the LUT is defined on that range.
The LUT is bilinear, so its derivative jumps at the bin knots. If only the
input check fails, re-run with another seed before suspecting the kernel.
Documentation: https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html
"""
from bilateral_filter_layer_4d_lut import BilateralFilterFunction4DLutGPU
import torch
from torch.autograd import gradcheck

SHAPE = (1, 1, 6, 8, 8, 8)  # [B, C, T, X, Y, Z]
WINDOW = (3, 3, 3, 3)
BINS = 8
NONDET_TOL = 1e-6           # kernel and LUT gradients use atomicAdd

torch.manual_seed(0)


def gaussian_1d(sigma, window):
    d = torch.arange(window, dtype=torch.double) - window // 2
    return torch.exp(-(d ** 2) / (2.0 * sigma ** 2))


def kernel(requires_grad=False):
    gt, gx, gy, gz = (gaussian_1d(1.0, w) for w in WINDOW)
    k = gt[:, None, None, None] * gx[None, :, None, None] * gy[None, None, :, None] * gz[None, None, None, :]
    return k.contiguous().cuda().requires_grad_(requires_grad)


def lut(requires_grad=False, sigma_r=0.3):
    grid = torch.linspace(0.0, 1.0, BINS, dtype=torch.double)
    table = torch.exp(-((grid[:, None] - grid[None, :]) ** 2) / (2.0 * sigma_r ** 2))
    return table.contiguous().cuda().requires_grad_(requires_grad)


def constant_input(layer_bf):
    tensor_in = torch.full(SHAPE, 0.5, dtype=torch.double, device='cuda')
    out = layer_bf(tensor_in, kernel(), lut())
    print(bool(torch.isfinite(out).all()) and torch.allclose(out, tensor_in, atol=1e-6))


def gradient_input(layer_bf):
    tensor_in = torch.rand(SHAPE, dtype=torch.double, device='cuda', requires_grad=True)
    print(gradcheck(layer_bf, (tensor_in, kernel(), lut()), eps=1e-6, atol=1e-5, nondet_tol=NONDET_TOL))


def gradient_kernel(layer_bf):
    tensor_in = torch.rand(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(layer_bf, (tensor_in, kernel(requires_grad=True), lut()), eps=1e-6, atol=1e-5, nondet_tol=NONDET_TOL))


def gradient_lut(layer_bf):
    tensor_in = torch.rand(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(layer_bf, (tensor_in, kernel(), lut(requires_grad=True)), eps=1e-6, atol=1e-5, nondet_tol=NONDET_TOL))


layer_bf_gpu = BilateralFilterFunction4DLutGPU.apply

print('-------------------------------------------------------------')
print('Gradcheck is passed if no error occurs and \'True\' is printed.')
print('-------------------------------------------------------------')
print('Constant input is reproduced:')
constant_input(layer_bf_gpu)

print('Gradient with respect to input:')
gradient_input(layer_bf_gpu)

print('Gradient with respect to kernel:')
gradient_kernel(layer_bf_gpu)

print('Gradient with respect to lut:')
gradient_lut(layer_bf_gpu)
