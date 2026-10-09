import itertools
import math

import torch
import torch.nn.functional as F

from bilateral_filter_layer_3d import BilateralFilter3d
from bilateral_filter_layer_4d import BilateralFilter4d
from bilateral_filter_layer_spt import BilateralFilter4dspt
from bilateral_filter_layer_nd import BilateralFilterNd


torch.manual_seed(0)
SHAPE = [1, 1, 10, 13, 17, 19]
TOL = 1e-6

def compare(name, old, new, x):
    old, new = old.cuda().double(), new.cuda().double()
    x_old = x.clone().requires_grad_()
    x_new = x.clone().requires_grad_()
    y_old, y_new = old(x_old), new(x_new)
    g = torch.randn_like(y_old)
    (y_old * g).sum().backward()
    (y_new * g).sum().backward()

    rows = [('output', (y_old - y_new).abs().max().item()),
            ('grad x', (x_old.grad - x_new.grad).abs().max().item())]
    new_params = dict(new.named_parameters())
    for pname, p in old.named_parameters():
        q = new_params[pname]
        rows.append((f'grad {pname}', abs(p.grad.item() - q.grad.item()) / max(abs(p.grad.item()), 1e-12)))

    ok = all(v < TOL for _, v in rows)
    print(f'{"OK  " if ok else "DIFF"} {name:38s} ' + '  '.join(f'{k} {v:.1e}' for k, v in rows))
    return ok


x = torch.rand(SHAPE, dtype=torch.double, device='cuda')
results = []
for s, r in itertools.product([0.3, 0.7, 1.0, 1.05, 1.5], [0.05, 0.4]):
    sx, sy, sz, st = s, 0.9 * s, 1.1 * s, 1.2 * s          # distinct per axis: catches swapped axes
    results.append(compare(f'3D       s={s:<4} r={r}',
                           BilateralFilter3d(sx, sy, sz, r),
                           BilateralFilterNd(r, axes=('x', 'y', 'z'), sigma_x=sx, sigma_y=sy, sigma_z=sz), x))
    results.append(compare(f'temporal s={s:<4} r={r}',
                           BilateralFilter4d(st, r, pad_t=4),
                           BilateralFilterNd(r, axes=('t',), sigma_t=st, pad_t=4), x))
    results.append(compare(f'SPT      s={s:<4} r={r}',
                           BilateralFilter4dspt(sx, sy, sz, st, r, pad_t=4),
                           BilateralFilterNd(r, axes=('t', 'x', 'y', 'z'),
                                             sigma_t=st, sigma_x=sx, sigma_y=sy, sigma_z=sz, pad_t=4), x))

print(f'\n{sum(results)}/{len(results)} comparisons equivalent within {TOL}')