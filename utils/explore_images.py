import nibabel as nib
import numpy as np
import torch

from utils.noise import SinogramNoise
from utils.rois import CALIBRATION_ROI, AORTA_ROI


def histogram():
    for path in ['/home/dpietsch/Data/Inhouse2/case_02/images/phase_05.nii',
                '/home/dpietsch/Data/Inhouse2/case_08/images/phase_05.nii']:
        img = nib.load(path)
        arr = np.asarray(img.dataobj)
        body = arr[(arr > -100) & (arr < 200)]
        hist, edges = np.histogram(body, bins=100)
        peak = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
        print(path, 'slope/inter', img.header.get_slope_inter(),
            'range', float(arr.min()), float(arr.max()), 'peak', float(peak))



def sinogram_noise():
    size = 512
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    disk = ((yy - size // 2) ** 2+ (xx - size // 2) ** 2) < (size // 3) ** 2
    x = torch.where(disk, (50.0 + 1024) / 4024, 0.0)[None].cuda()

    noiser = SinogramNoise(dosage=0.25, i0_full=1e5)
    recon, _ = noiser(x)
    to_hu = lambda v: v * 4024 - 1024
    print('in ', to_hu(x[0][disk]).mean().item())
    print('out ', to_hu(recon[0][disk]).mean().item())

def test_sinogram_noise():
    size = 512
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    disk = ((yy - size // 2) ** 2 + (xx - size // 2) ** 2) < (size // 3) ** 2
    to_norm = lambda hu: (hu + 1024.0) / 4024.0
    to_hu = lambda v: v * 4024.0 - 1024.0

    for beam in ('Parallel-Beam',):
        noiser = SinogramNoise(b_type=beam, image_size=size, dosage=0.25, i0_full=1e5)
        for density_hu in (50.0, 1000.0):
            x = torch.where(disk, to_norm(density_hu), to_norm(-1024.0))[None].cuda()
            # Call the internals: clamp(0, 1) in __call__ would hide a negative offset.
            clean, _ = noiser(x)
            print(f'{beam:14s} {density_hu:7.1f} HU  '
                f'inside {to_hu(clean[0][disk]).mean().item():8.1f}  '
                f'outside {to_hu(clean[0][~disk]).mean().item():8.1f}')


def check_std_roi():
    root = '/home/dpietsch/Data/.cache'
    clean_std, noisy_std = [], []
    for data, rois in [('Inhouse1Processed_v3', CALIBRATION_ROI), ('Inhouse2Processed_v3', AORTA_ROI)]:
        for scan, (d0, d1, r0, r1, c0, c1) in rois.items():
            try:
                clean = np.load(f'{root}/{data}/{scan}/images/phase_05.npy')[0, d0:d1, r0:r1, c0:c1] * 4024.0
                noisy = np.load(f'{root}/{data}/{scan}/noise/noise_phase_05.npy')[0, d0:d1, r0:r1, c0:c1] * 4024.0
            except FileNotFoundError:
                continue
            clean_std.append(clean.std())
            noisy_std.append(noisy.std())
            print(f'{scan:40s} clean {clean.std():5.1f} HU    noisy {noisy.std():6.1f} HU')

    clean_std = np.array(clean_std, dtype=np.float32)
    clean_std_mean = clean_std.mean()
    clean_std_std = clean_std.std()

    noisy_std = np.array(noisy_std, dtype=np.float32)
    noisy_std_mean = noisy_std.mean()
    noisy_std_std = noisy_std.std()
    print(f'Over all scans: mean clean {clean_std_mean:5.1f} HU   std clean {clean_std_std:6.1f}  '
          f'\n--------------: mean noisy {noisy_std_mean:5.1f} HU   std noisy {noisy_std_std:6.1f}')

check_std_roi()

