import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import nibabel as nib

from data.dataset import CTValidationDataset

from test_models import HU_MIN, HU_MAX, HU_RANGE, PHASE_INDEX
from utils.rois import AORTA_ROI, CALIBRATION_ROI

DATA = {
    'train': ('/home/dpietsch/Data/.cache/Inhouse1Processed_v3', '/home/dpietsch/Data/Inhouse1', CALIBRATION_ROI),
    'test':  ('/home/dpietsch/Data/.cache/Inhouse2Processed_v3', '/home/dpietsch/Data/Inhouse2', AORTA_ROI),
} 

OUT = Path('explorations/roi_picking')
ROI_H, ROI_W = 12, 12
SLAB = 8
N_CANDIDATES = 6
SCALE, GRID = 2, 32
LUNG_MIN_FRACTION = 0.5
SPINE_HU = 200.0
MAX_SPINE_DIST = 45
ANTERIOR_MARGIN = 8

SOFT_CENTER, SOFT_WIDTH = 40.0, 400.0
SOFT_LO, SOFT_HI = SOFT_CENTER - SOFT_WIDTH / 2, SOFT_CENTER + SOFT_WIDTH / 2

TISSUE_LO_HU, TISSUE_HI_HU = 0.0, 80.0
MAX_SEARCH_SCORE = 15.0


def spine_centre(slab_hu):
    body = slab_hu > -950.0
    rows, cols = np.where(body.any(axis=1))[0], np.where(body.any(axis=0))[0]
    if rows.size == 0:
        return None
    row_lo = rows[0] + (rows[-1] - rows[0]) / 3
    row_hi = rows[0] + 2 * (rows[-1] - rows[0]) / 3
    col_hi = cols[0] + 0.5 * (cols[-1] - cols[0])

    bone = (slab_hu > SPINE_HU).astype(np.uint8)
    if bone.sum() < 50:
        return None
    n, _, stats, centroids = cv2.connectedComponentsWithStats(bone, 8)
    best, best_area = None, 0
    for i in range(1, n):
        col, row = centroids[i]
        if not (row_lo <= row <= row_hi) or col > col_hi:
            continue
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_area:
            best, best_area = (float(row), float(col)), area
    return best


def load_phase_hu(row, phase=PHASE_INDEX):
    path = row['clean'][phase]
    if '.npy' in path:
        vol = np.squeeze(np.load(path).astype(np.float32), axis=0)
        vol = vol * HU_RANGE + HU_MIN
    elif '.nii' in row['clean'][phase]:
        vol = nib.load(path)
        vol = np.asarray(vol.dataobj).astype(np.float32)
        vol = vol.transpose(2, 0, 1)
    else:
        raise ValueError("Image format not supported")
    
    return vol

def lung_slices(vol_hu, min_fraction=LUNG_MIN_FRACTION):
    lung = (vol_hu > -950) & (vol_hu < -500)
    area = lung.reshape(lung.shape[0], -1).sum(axis=1)
    if area.max() == 0:
        return None
    keep = np.where(area >= min_fraction * area.max())[0]
    return int(keep[0]), int(keep[-1]) + 1


def local_stats(slice_hu, h, w):
    mean = cv2.blur(slice_hu, (w, h), borderType=cv2.BORDER_REPLICATE)
    mean_sq = cv2.blur(slice_hu * slice_hu, (w, h), borderType=cv2.BORDER_REPLICATE)
    return mean, np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))

def propose(vol_hu, n=N_CANDIDATES):
    """proposes most homogeneous soft tissue boxes, as (score, d, r0, c0)"""
    D, H, W = vol_hu.shape
    bounds = lung_slices(vol_hu)
    if bounds is None:
        return []
    d_lo, d_hi = bounds
    found = []
    for d in range(d_lo, max(d_hi - SLAB, d_lo) + 1, max(SLAB // 2, 1)):
        if d + SLAB > D:
            break
        slab = vol_hu[d:d + SLAB].mean(axis=0)
        centre = spine_centre(slab)
        if centre is None:
            continue
        mean, std = local_stats(slab, ROI_H, ROI_W)

        row_c, col_c = centre
        rr, cc = np.ogrid[:H, :W]
        near_spine = ((rr - row_c) ** 2 + (cc - col_c) ** 2) <= MAX_SPINE_DIST ** 2
        anterior = cc > (col_c + ANTERIOR_MARGIN)
        inside = (mean > TISSUE_LO_HU) & (mean < TISSUE_HI_HU)
        margin_h, margin_w = ROI_H // 2 + 1, ROI_W // 2 + 1
        border = np.zeros_like(inside)
        border[margin_h:H - margin_h, margin_w:W - margin_w] = True

        valid = inside & border & near_spine & anterior & (std < MAX_SEARCH_SCORE)
        score = np.where(valid, std, np.inf)
        for _ in range(3):
            idx = int(np.argmin(score))
            r, c = divmod(idx, W)
            if not np.isfinite(score[r, c]):
                break
            
            found.append((float(score[r, c]), d, r - ROI_H // 2, c - ROI_W // 2))
            score[max(r - ROI_H, 0):r + ROI_H, max(c - ROI_W, 0):c + ROI_W] = np.inf

    found.sort(key=lambda t: t[0])
    kept = []
    for cand in found:
        if all(abs(cand[1] - k[1]) > SLAB or
               abs(cand[2] - k[2]) > ROI_H or
               abs(cand[3] - k[3]) > ROI_W for k in kept):
            
            kept.append(cand)
        if len(kept) == n:
            break
    return kept

def render(slab_hu, boxes, path, labels=None):
    img = ((slab_hu - SOFT_LO) / (SOFT_HI - SOFT_LO)).clip(0, 1)
    img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    H, W = slab_hu.shape

    for c in range(0, W, GRID):
        cv2.line(img, (c * SCALE, 0), (c * SCALE, H * SCALE), (60, 60, 60), 1)
        cv2.putText(img, str(c), (c * SCALE + 2, 12), cv2.FONT_HERSHEY_PLAIN, 0.7, (120, 120, 120), 1)

    for r in range(0, H, GRID):
        cv2.line(img, (0, r * SCALE), (W * SCALE, r * SCALE), (60, 60, 60), 1)
        cv2.putText(img, str(r), (2, r * SCALE + 12), cv2.FONT_HERSHEY_PLAIN, 0.7, (120, 120, 120), 1)

    for i, (r0, c0) in enumerate(boxes):
        cv2.rectangle(img, (c0 * SCALE, r0 * SCALE),
                      ((c0 + ROI_W) * SCALE, (r0 + ROI_H) *SCALE),
                      (255, 165, 0), 2)
        text = labels[i] if labels else str(i)
        cv2.putText(img, text, (c0 * SCALE, r0 * SCALE - 4),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 165, 0), 2)

    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def hu_landmarks(vol_hu):
    air = float(np.percentile(vol_hu, 1))
    body = vol_hu[(vol_hu > -100) & (vol_hu < 200)]
    hist, edges = np.histogram(body, bins=100)
    peak = float(0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1]))
    return air, peak

def soft_tissue_mode(vol_hu, lo=-50.0, hi=150.0, bins=200):
    """Mode of the soft tissue distribution.
    The window starts above fat so the estimate cannot slide onto the fat peak,
    and the bin edges are fixed rather than taken from the data, so the number
    is comparable between scans and between source and cache.
    """
    body = vol_hu[(vol_hu > lo) & (vol_hu < hi)]
    if body.size == 0:
        return float('nan')
    hist, edges = np.histogram(body, bins=bins, range=(lo, hi))
    return float(0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1]))

def tissue_modes(vol_hu):
    fat = soft_tissue_mode(vol_hu, lo=-150.0, hi=-50.0, bins=100)
    soft = soft_tissue_mode(vol_hu, lo=0.0, hi=150.0, bins=150)
    return fat, soft

def histogram_peaks(vol_hu, lo=-250.0, hi=250.0, bins=250, smooth=9, top=3):
    vals = vol_hu[(vol_hu > lo) & (vol_hu < hi)]
    hist, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    h = np.convolve(hist, np.ones(smooth) / smooth, mode='same')
    centres = 0.5 * (edges[:-1] + edges[1:])
    peaks = [i for i in range(1, len(h) - 1) if h[i] > h[i - 1] and h[i] >= h[i + 1]]
    peaks.sort(key=lambda i: h[i], reverse=True)
    return [(float(centres[i]), float(h[i] / h.max())) for i in peaks[:top]]

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', choices=DATA, default='train',
                        help='train picks calibration ROIs, test picks evaluation ROIs')
    parser.add_argument('--hist', action='store_true', help='calculate histogram statistics')
    parser.add_argument('--scan', default=None, help='only this scan id')
    parser.add_argument('--verify', nargs=6, type=int, default=None,
                        metavar=('D0', 'D1', 'R0', 'R1', 'C0', 'C1'),
                        help='render one chosen box and print its statistics')
    parser.add_argument('--verify-table', action='store_true',
                        help='check every ROI already recorded for this dataset')
    args = parser.parse_args()

    cache, source, table = DATA[args.data]
    out = OUT / args.data
    out.mkdir(parents=True, exist_ok=True)
    df = CTValidationDataset.generate_dataframe(cache)
    
    if args.scan:
        df = df.loc[[args.scan]]

    if args.hist:
        df_orig = CTValidationDataset.generate_origdata_dataframe(source)
        for scan_id, row in df.iterrows():
            peaks = histogram_peaks(load_phase_hu(df_orig.loc[scan_id]))
            print(f'{scan_id:40s}   ' + '  '.join(f'{hu:7.1f} HU ({r:.2f})' for hu, r in peaks))
        return


    rows = []
    for scan_id, row in df.iterrows():
        vol = load_phase_hu(row)

        if args.verify or args.verify_table:
            if args.verify_table and scan_id not in table:
                continue

            d0, d1, r0, r1, c0, c1 = args.verify or table[scan_id]
            roi = vol[d0:d1, r0:r1, c0:c1]
            print(f'{scan_id}: mean {roi.mean():7.1f} HU   std {roi.std():6.1f} HU    '
                  f'range [{roi.min():.0f}, {roi.max():.0f}]')
            render(vol[d0:d1].mean(axis=0), [(r0, c0)],
                   out / f'{scan_id}_verify.png', labels=['chosen'])
            continue

        candidates = propose(vol)
        if not candidates:
            print(f'{scan_id}: no soft tissue candidate found')
            continue

        best_d = candidates[0][1]
        on_slab = [(r0, c0) for _, d, r0, c0 in candidates if abs(d - best_d) <= SLAB]
        render(vol[best_d:best_d + SLAB].mean(axis=0), on_slab,
               out / f'{scan_id}_candidates.png',
               labels=[str(i) for i in range(len(on_slab))])

        for i, (score, d, r0, c0) in enumerate(candidates):
            roi = vol[d:d + SLAB, r0:r0 + ROI_H, c0:c0 + ROI_W]
            rows.append(dict(scan=scan_id, candidate=i,
                             search_score=score,
                             std_hu=float(roi.std()),
                             mean_hu=float(roi.mean()),
                             d0=d, d1=d + SLAB, r0=r0, r1=r0 + ROI_H,
                             c0=c0, c1=c0 + ROI_W))

        print(f'{scan_id}: {vol.shape[0]} slices, best std {candidates[0][0]:.1f} HU at slab {best_d}')

    if rows:
        path = out / 'roi_candidates.csv'
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f'\nwrote {path} and {out}/*_candidates.png')


if __name__ == '__main__':
    main()
