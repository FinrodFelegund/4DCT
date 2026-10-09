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

WINDOW = (5, 3, 1, 3)                     # every axis different, including size 1
SHAPE = (1, 1, 7, 6, 5, 8)                # [B, C, T, X, Y, Z], all sizes different


def reference(x, k, s):
    x = x[0, 0]
    halves = [n // 2 for n in k.shape]
    num, den = torch.zeros_like(x), torch.zeros_like(x)
    for idx in torch.cartesian_prod(*[torch.arange(n) for n in k.shape]).tolist():
        delta = [i - h for i, h in zip(idx, halves)]
        src = tuple(slice(max(0, d), n + min(0, d)) for d, n in zip(delta, x.shape))
        dst = tuple(slice(max(0, -d), n - max(0, d)) for d, n in zip(delta, x.shape))
        nb, home = x[src], x[dst]
        w = k[tuple(idx)] * torch.exp(-(nb - home) ** 2 / (2 * s ** 2))
        num[dst] += w * nb
        den[dst] += w
    return (num / den)[None, None]


torch.manual_seed(0)
x = torch.rand(SHAPE, dtype=torch.double, device='cuda')
k = torch.rand(WINDOW, dtype=torch.double, device='cuda') + 0.1
s = torch.tensor(0.3, dtype=torch.double, device='cuda')

print('forward max |cuda - reference|:', (BilateralFilterFunction4dlearnablesptGPU.apply(x, k, s) - reference(x, k, s)).abs().max().item())
print('gradcheck input :', gradcheck(BilateralFilterFunction4dlearnablesptGPU.apply, (x.clone().requires_grad_(), k, s), eps=1e-6, atol=1e-5, nondet_tol=1e-6))
print('gradcheck kernel:', gradcheck(BilateralFilterFunction4dlearnablesptGPU.apply, (x, k.clone().requires_grad_(), s), eps=1e-6, atol=1e-5, nondet_tol=1e-6))
