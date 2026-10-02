import os
from glob import glob
from tqdm import tqdm
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from utils.pick_aorta import soft_tissue_mode, DATA
from data.dataset import CTValidationDataset
from test_models import HU_MIN, HU_RANGE, PHASE_INDEX

DATASETS = ['/home/dpietsch/Data/Inhouse1', '/home/dpietsch/Data/Inhouse2']
TARGET_Z_MM = 2.5
MIN_SLICES = 64


def lateral_edge_tissue(img, thresh=-500.0):
    arr = np.asarray(img.dataobj)
    lr = next(i for i, c in enumerate(nib.aff2axcodes(img.affine)) if c in 'LR')
    first = np.take(arr, [0, 1], axis=lr)
    last = np.take(arr, [-2, -1], axis=lr)
    return max(float((first > thresh).mean()), float((last > thresh).mean()))

def audit_scans():
    rows = []
    for root in DATASETS:
        for scan in tqdm(sorted(os.listdir(root))):
            n_z, dz, dxy, matrix = [], [], [], []
            for path in sorted(glob(f'{root}/{scan}/images/phase_*.nii*')):
                img = nib.load(path)
                codes = nib.aff2axcodes(img.affine)
                z = next(i for i, c in enumerate(codes) if c in 'SI')
                inplane = [i for i in range(3) if i != z]
                n_z.append(img.shape[z])
                zooms = img.header.get_zooms()[:3]
                dz.append(float(zooms[z]))
                dxy.append(tuple(round(float(zooms[i]), 3) for i in inplane))
                matrix.append(tuple(int(img.shape[i]) for i in inplane))
                if os.path.basename(path).startswith('phase_05'):
                    edge = lateral_edge_tissue(img)
                    mode = soft_tissue_mode(np.asarray(img.dataobj), lo=-1350.0, hi=150.0)
            coverage = min(n_z) * dz[0] if n_z else 0.0
            fov = tuple(round(n * d, 1) for n, d in zip(matrix[0], dxy[0])) if matrix else None
            rows.append(dict(dataset=os.path.basename(root), scan=scan, phases=len(n_z),
                            slices_min=min(n_z, default=0), slices_max=max(n_z, default=0),
                            spacing_z=round(dz[0], 2) if dz else float('nan'),
                            spacing_xy=dxy[0] if dxy else float('nan'),
                            matrix_xy=matrix[0] if matrix else None,
                            fov_mm=fov,
                            coverage_mm=round(coverage, 1),
                            slices_resampled=int(coverage // TARGET_Z_MM),
                            lateral_edge_tissue=round(edge, 3),
                            lung_window_mode=round(mode, 3)))

    df = pd.DataFrame(rows)
    df['flag'] = ''
    df.loc[df['phases'] != 10, 'flag'] += 'phases '
    df.loc[df['slices_min'] != df['slices_max'], 'flag'] += 'uneven '
    df.loc[df['slices_resampled'] < MIN_SLICES, 'flag'] += 'short '
    df.loc[df['lateral_edge_tissue'] > 0.2, 'flag'] += 'truncated'
    print(df.to_string(index=False))
    os.makedirs('explorations', exist_ok=True)
    df.to_csv('explorations/scan_audit.csv', index=False)

def audit_uneven_scans():
    uneven_scans = [glob('/home/dpietsch/Data/Inhouse1/069_*/images/phase_*.nii'), glob('/home/dpietsch/Data/Inhouse1/558_Lunge_*/images/phase_*.nii')]

    for scan in uneven_scans:
        for p in sorted(scan):
            img = nib.load(p)
            print(p.split('/')[-1], img.shape, np.round(img.affine[:3, 3], 2))

def check_missing_slice_position():
    uneven_scans = ['/home/dpietsch/Data/Inhouse1/069_4DCT_Lunge_amplitudebased_complete/images/phase_06.nii', '/home/dpietsch/Data/Inhouse1/558_Lunge_amplitudebased/images/phase_02.nii']
    neighbours =  ['/home/dpietsch/Data/Inhouse1/069_4DCT_Lunge_amplitudebased_complete/images/phase_07.nii', '/home/dpietsch/Data/Inhouse1/558_Lunge_amplitudebased/images/phase_03.nii']

    for a_scan, b_scan in zip(uneven_scans, neighbours):
        a = np.asarray(nib.load(a_scan).dataobj)
        b = np.asarray(nib.load(b_scan).dataobj)
        print(f'scan: {a_scan}')
        same = np.array([np.abs(a[..., k] - b[..., k]).mean() for k in range(a.shape[2])])
        shifted = np.array([np.abs(a[..., k] - b[..., k + 1]).mean() for k in range(a.shape[2])])
        print((shifted < same).astype(int))


def generate_processed_niifiles():
    SPACING = (1.16, 1.16, 2.5)
    df = CTValidationDataset.generate_dataframe(DATA['test'][0])
    for scan_id, row in tqdm(df.iterrows()):
        vol = np.squeeze(np.load(row.loc['clean'][PHASE_INDEX]).astype(np.float32), axis=0)
        hu = np.round(vol * HU_RANGE + HU_MIN).astype(np.int16)
        img = nib.Nifti1Image(hu.transpose(1, 2, 0), affine=np.diag([*SPACING, 1.0]))
        out = Path('explorations/viewer') / f'test/{row['scan']}/images/phase_{PHASE_INDEX:02d}.nii'
        out.parent.mkdir(parents=True, exist_ok=True)
        nib.save(img, str(out))
        print(f'wrote {out}')

generate_processed_niifiles()






