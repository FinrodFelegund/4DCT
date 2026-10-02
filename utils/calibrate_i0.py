import argparse
import csv
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from monai.transforms import Compose
import cv2

from data.dataset import CTCacheDataset, EXCLUDED
from utils.noise import SinogramNoise
from utils.rois import AORTA_ROI, CALIBRATION_ROI

PHASE_INDEX = 5
DOSAGE = 0.25
TARGET_HU = 108.0
I0_GRID = np.geomspace(1e4, 2e6, 20)
SOURCES = {
    'train': ('/home/dpietsch/Data/Inhouse1', CALIBRATION_ROI),
    'test': ('/home/dpietsch/Data/Inhouse2', AORTA_ROI),
}

def to_hu(v):
    return v * 4024.0 - 1024.0

def fit_i0(i0_grid, stds, target=TARGET_HU):
    x, y, t = np.log(np.asarray(i0_grid, float)), np.log(np.asarray(stds, float)), np.log(target)
    for i in range(len(x) - 1):
        if (y[i] - t) * (y[i + 1] - t) <= 0:
            w = (t - y[i]) / (y[i + 1] - y[i])
            return float(np.exp(x[i] + w * (x[i + 1] - x[i])))

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', choices=SOURCES, default='train')
    args = parser.parse_args()
    source, rois = SOURCES[args.data]

    df = CTCacheDataset.generate_dataframe(source, '/UNUSED')
    pre = Compose(CTCacheDataset.get_cache_transforms(dosage=DOSAGE, i0_full=0.0).transforms[:-2])

    rows = []
    for scan, (d0, d1, r0, r1, c0, c1) in tqdm(rois.items()):
        if scan in EXCLUDED:
            continue
        data = CTCacheDataset(df.loc[[scan]], pre)[0]
        slab = torch.as_tensor(data[f'phase_0{PHASE_INDEX}'][0][d0:d1]).float().cuda()
        roi = (slice(None), slice(r0, r1), slice(c0, c1))

        stds = []
        for i0 in I0_GRID:
            torch.manual_seed(0)
            clean, noisy = SinogramNoise(dosage=DOSAGE, i0_full=i0)(slab)
            stds.append(to_hu(noisy[roi]).std().item())

        clean_std = to_hu(clean[roi]).std().item()
        i0_fit = fit_i0(I0_GRID, stds, TARGET_HU)
        rows.append(dict(scan=scan, clean_std_hu=round(clean_std, 2),
                         i0_fit='' if i0_fit is None else round(i0_fit)))

        print(f'{scan:40s} clean {clean_std:5.1f} HU   '
              f'i0 {"outside grid" if i0_fit is None else f"{i0_fit:10.0f}"}   '
              f'noisy {stds[0]:6.1f} to {stds[-1]:6.1f} HU across the grid')

    out = Path(f'explorations/i0_{args.data}.csv')
    out.parent.mkdir(exist_ok=True, parents=True)
    with open(out, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['scan', 'clean_std_hu', 'i0_fit'])
        writer.writeheader()
        writer.writerows(rows)

    fitted = np.array([r['i0_fit'] for r in rows if r['i0_fit'] != ''], float)
    missing = [r['scan'] for r in rows if r['i0_fit'] == '']
    print(f'\nwrote {out}: median i0 {np.median(fitted):.0f},  '
          f'range {fitted.min():.0f} to {fitted.max():.0f}')
    if missing:
        print(f'target outside the grid for: {missing}')

def compare_with_cache(scan='case_07', cache_dir='/home/dpietsch/Data/.cache/Inhouse2Processed_v3'):
    df = CTCacheDataset.generate_dataframe(SOURCES['test'][0], 'unused')
    pre = Compose(CTCacheDataset.get_cache_transforms(dosage=DOSAGE, i0_full=0.0).transforms[:-2])
    vol = torch.as_tensor(CTCacheDataset(df.loc[[scan]], pre)[0][f'phase_0{PHASE_INDEX}'][0]).float()
    model = SinogramNoise(dosage=DOSAGE, i0_full=1e5)
    clean = torch.cat([model(vol[s:s+16].cuda())[0].cpu() for s in range(0, len(vol), 16)])

    cached = np.squeeze(np.load(f'{cache_dir}/{scan}/images/phase_0{PHASE_INDEX}.npy'), axis=0)
    print(f'{scan}: pipeline {tuple(clean.shape)}  cache {cached.shape}')
    if tuple(clean.shape) == cached.shape:
        a, b = clean.numpy(), cached
        mid = slice(len(a) // 2 - 8, len(a) // 2 + 8)       # 16 central slices, for speed

        shifts = []
        for dd in (-2, -1, 0, 1, 2):
            for dr in (-2, -1, 0, 1, 2):
                for dc in (-2, -1, 0, 1, 2):
                    rolled = np.roll(b, (dd, dr, dc), axis=(0, 1, 2))
                    shifts.append((np.abs(a[mid] - rolled[mid]).mean() * 4024.0, (dd, dr, dc)))
        shifts.sort()
        print('  best shifts (mean |diff| in HU, (d, r, c)):', [(round(v, 2), s) for v, s in shifts[:3]])

        for name, flipped in [('flip d', b[::-1]), ('flip r', b[:, ::-1]), ('flip c', b[:, :, ::-1])]:
            print(f'  {name}: mean |diff| {np.abs(a[mid] - flipped[mid]).mean() * 4024.0:.2f} HU')

        s = len(a) // 2                                       # mid grey = no difference
        img = ((a[s] - b[s]) * 4024.0).clip(-200, 200)
        cv2.imwrite('explorations/cache_diff.png', ((img + 200) / 400 * 255).astype(np.uint8))


if __name__ == '__main__':
    main()