"""
Gradcheck for the bilateral filter with learnable 4D spatial kernel
(bilateral_filter_layer_learnable_spt), GPU only.
Documentation: https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html
"""
from bilateral_filter_layer_learnable_spt import BilateralFilterFunction4dlearnablesptGPU, gaussian_4d
import torch
from torch.autograd import gradcheck

SHAPE = (1, 1, 6, 8, 8, 8)  # [B, C, T, X, Y, Z]
WINDOW = (3, 3, 3, 3)       # small kernel keeps the check fast
NONDET_TOL = 1e-6           # kernel gradient uses atomicAdd


def kernel(requires_grad=False):
    k = gaussian_4d(1.0, 1.0, 1.0, 1.0, WINDOW, dtype=torch.double).cuda()
    return k.requires_grad_(requires_grad)


def color_sigma(requires_grad=False):
    return torch.tensor(0.5, dtype=torch.double, requires_grad=requires_grad)


def constant_input(layer_bf):
    tensor_in = torch.full(SHAPE, 0.5, dtype=torch.double, device='cuda')
    out = layer_bf(tensor_in, kernel(), color_sigma())
    print(bool(torch.isfinite(out).all()) and torch.allclose(out, tensor_in, atol=1e-6))


def gradient_input(layer_bf):
    tensor_in = torch.randn(SHAPE, dtype=torch.double, device='cuda', requires_grad=True)
    print(gradcheck(layer_bf, (tensor_in, kernel(), color_sigma()), eps=1e-6, atol=1e-5, nondet_tol=NONDET_TOL))


def gradient_kernel(layer_bf):
    tensor_in = torch.randn(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(layer_bf, (tensor_in, kernel(requires_grad=True), color_sigma()), eps=1e-6, atol=1e-5, nondet_tol=NONDET_TOL))


def gradient_r(layer_bf):
    tensor_in = torch.randn(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(layer_bf, (tensor_in, kernel(), color_sigma(requires_grad=True)), eps=1e-3, atol=1e-3, nondet_tol=NONDET_TOL))


layer_bf_gpu = BilateralFilterFunction4dlearnablesptGPU.apply

print('-------------------------------------------------------------')
print('Gradcheck is passed if no error occurs and \'True\' is printed.')
print('-------------------------------------------------------------')
print('Constant input is reproduced:')
constant_input(layer_bf_gpu)

print('Gradient with respect to input:')
gradient_input(layer_bf_gpu)

print('Gradient with respect to kernel:')
gradient_kernel(layer_bf_gpu)

print('Gradient with respect to sigma_range:')
gradient_r(layer_bf_gpu)
