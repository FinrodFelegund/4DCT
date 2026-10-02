import itertools
import numpy as np
import torch
from tqdm import tqdm

from utils.processing import interior

@torch.no_grad()
def sweep_sigmas(model, loader, spatial_values, range_values, margin=8, device='cuda'):
    print('Sweeping sigmas')
    spatial = [p for n, p in model.named_parameters()
               if n.rsplit('.', 1)[-1] in ('sigma_x', 'sigma_y', 'sigma_z')]

    ranges = [p for n, p in model.named_parameters()
              if n.endswith('color_sigma')]
    saved = [p.clone() for p in spatial + ranges]

    def val_mse():
        return np.mean([torch.nn.functional.mse_loss(interior(model(noisy.to(device)), margin),
                                                     interior(clean.to(device), margin)).item()
                        for clean, noisy in tqdm(loader)])

    print(f'learned parameters: validation MSE {val_mse():.4e}')
    for s, r in itertools.product(spatial_values, range_values):
        for p in spatial:
            p.fill_(s)
        for p in ranges:
            p.fill_(r)

        print(f'sigma {s:.2f} vox sigma_r {r:.4f} ({r * 4024:5.0f} HU)   validation MSE {val_mse():.4e}')

    for p, v in zip(spatial + ranges, saved):
        p.copy_(v)

