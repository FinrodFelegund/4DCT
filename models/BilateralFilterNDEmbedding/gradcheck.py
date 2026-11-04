"""
Author: Daniel Pietsch
Email: daniel.pietsch@studium.uni-hamburg.de
"""

from bilateral_filter_layer_nd_embedding import BilateralFilterFunctionNdEmbeddingGPU, pad_circular_t, unpad_circular_t
import torch
from torch.autograd import gradcheck

SHAPE = (1, 1, 1, 5, 7, 9)
VALUES = {'sigma_t': 0.5,'sigma_x': 0.5, 'sigma_y': 0.5, 'sigma_z': 0.5}
PAD = 4
H, L = 8, 1
ACTIVATION = 1
NONDET_TOL = 1e-6

def sigmas(grad=None):
    return tuple(torch.tensor(v, dtype=torch.double, device='cuda', requires_grad=(k==grad)) for k, v in VALUES.items())

torch.manual_seed(0)

@torch.no_grad()
def embedding(requires_grad=False):
    range_net = torch.nn.Sequential(
        torch.nn.Linear(4, H, dtype=torch.double),
        *[torch.nn.Linear(H, H, dtype=torch.double) for _ in range(L)],
        torch.nn.Linear(H, 1, dtype=torch.double),
    )

    first, hidden, last = range_net[0], range_net[1:-1], range_net[-1]
    parts = [first.weight.reshape(-1), first.bias]
    parts += [torch.stack([l.weight for l in hidden]).reshape(-1),
            torch.stack([l.bias for l in hidden]).reshape(-1)]
    parts += [last.weight.reshape(-1), last.bias]

    parts = torch.cat(parts).contiguous().cuda().requires_grad_(requires_grad)
    return parts

def padded(layer_bf):
    def f(x, sigma_t, sigma_x, sigma_y, sigma_z, embedding):
        out = layer_bf(pad_circular_t(x, PAD), sigma_t, sigma_x, sigma_y, sigma_z, embedding, H, L, ACTIVATION, PAD)
        return unpad_circular_t(out, PAD)
    return f


def constant_input(layer_bf):
    tensor_in = torch.full(SHAPE, 0.5, dtype=torch.double, device='cuda')
    out = padded(layer_bf)(tensor_in, *sigmas(), embedding())
    print(bool(torch.isfinite(out).all()) and torch.allclose(out, tensor_in, atol=1e-6))

def gradient_input(layer_bf):
    tensor_in = torch.rand(SHAPE, dtype=torch.double, device='cuda', requires_grad=True)
    print(gradcheck(padded(layer_bf), (tensor_in, *sigmas(), embedding()), eps=1e-6, atol=1e-5, nondet_tol=NONDET_TOL, fast_mode=True))

def gradient_t(layer_bf):
    tensor_in = torch.rand(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(padded(layer_bf), (tensor_in, *sigmas('sigma_t'), embedding()), eps=1e-2, atol=1e-5, nondet_tol=NONDET_TOL))

def gradient_x(layer_bf):
    tensor_in = torch.rand(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(padded(layer_bf), (tensor_in, *sigmas('sigma_x'), embedding()), eps=1e-2, atol=1e-5, nondet_tol=NONDET_TOL))

def gradient_y(layer_bf):
    tensor_in = torch.rand(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(padded(layer_bf), (tensor_in, *sigmas('sigma_y'), embedding()), eps=1e-2, atol=1e-5, nondet_tol=NONDET_TOL))

def gradient_z(layer_bf):
    tensor_in = torch.rand(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(padded(layer_bf), (tensor_in, *sigmas('sigma_z'), embedding()), eps=1e-2, atol=1e-5, nondet_tol=NONDET_TOL))

def gradient_embedding(layer_bf):
    tensor_in = torch.rand(SHAPE, dtype=torch.double, device='cuda')
    print(gradcheck(padded(layer_bf), (tensor_in, *sigmas(), embedding(True)), eps=1e-6, atol=1e-5, nondet_tol=NONDET_TOL))

layer_bf_gpu = BilateralFilterFunctionNdEmbeddingGPU.apply

print('-------------------------------------------------------------')
print('Gradcheck is passed if no error occurs and \'True\' is printed.')
print('-------------------------------------------------------------')
print('Constant input is reproduced:')
constant_input(layer_bf_gpu)

print('Gradient with respect to input')
gradient_input(layer_bf_gpu)

print('Gradient with respect to sigma t:')
gradient_t(layer_bf_gpu)

print('Gradient with respect to sigma x:')
gradient_x(layer_bf_gpu)

print('Gradient with respect to sigma y:')
gradient_y(layer_bf_gpu)

print('Gradient with respect to sigma z:')
gradient_z(layer_bf_gpu)

print('Gradient with respetct to embedding:')
gradient_embedding(layer_bf_gpu)