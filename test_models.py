import argparse
import yaml
from pathlib import Path

from cv2 import cvtColor, COLOR_GRAY2RGB, rectangle
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_fn
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
from tqdm import tqdm
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt

from data.dataset import CTValidationDataset
from models.filterbank import FilterBank
from utils.processing import interior, inplane_interior, body_box
from utils.rois import AORTA_ROI


DEVICE = 'cuda'
RESULTS = Path('results')
TEST_DATA = '/home/dpietsch/Data/.cache/Inhouse2Processed_v3'
DENOISE_DIR = RESULTS / 'pred'
DENOISE_ROW = 'denoised'

HU_MIN, HU_MAX = -1024.0, 3000.0
HU_RANGE = HU_MAX - HU_MIN

LUNG_CENTER, LUNG_WIDTH = -500.0, 1500.0
LUNG_LO = LUNG_CENTER - (LUNG_WIDTH / 2)
LUNG_HI = LUNG_CENTER + (LUNG_WIDTH / 2)

PHASE_INDEX = 5
CROP_DY, CROP_DX = 110, 120
MARGIN = 8
MOTION_QUANTILE = 0.9

MODELS = {
    'BilateralFilter3D': {
        'parameters': '/home/dpietsch/4DCT/parameters/BilateralFilter3D_epoch52_loss0.000069.pth',
        'config': '/home/dpietsch/4DCT/configs/BilateralFilter3D.yml',
    },
    'BilateralFilter4D': {
        'parameters': '/home/dpietsch/4DCT/parameters/BilateralFilter4D_epoch59_loss0.000055.pth',
        'config': '/home/dpietsch/4DCT/configs/BilateralFilter4D.yml',
    },
    'BilateralFilterSPT4D': {
        'parameters': '/home/dpietsch/4DCT/parameters/BilateralFilterSPT4D_epoch59_loss0.000056.pth',
        'config': '/home/dpietsch/4DCT/configs/BilateralFilterSPT4D.yml',
    },
    'BilateralFilterSPTWeightedCenter4D': {
        'parameters': '/home/dpietsch/4DCT/parameters/BilateralFilterSPTWeightedCenter4D_epoch55_loss0.000053.pth',
        'config': '/home/dpietsch/4DCT/configs/BilateralFilterSPTWeightedCenter4D.yml',
    },
    'LearnableFilter4D': {
        'parameters': '/home/dpietsch/4DCT/parameters/LearnableFilter4D_epoch29_loss0.000052.pth',
        'config': '/home/dpietsch/4DCT/configs/LearnableFilter4D.yml',
    },
    'LearnableFilterSPT4D': {
        'parameters': '/home/dpietsch/4DCT/parameters/LearnableFilterSPT4D_epoch29_loss0.000053.pth',
        'config': '/home/dpietsch/4DCT/configs/LearnableFilterSPT4D.yml',
    },
    'LearnableFilter4DLUT': {
        'parameters': '/home/dpietsch/4DCT/parameters/LearnableFilter4DLUT_epoch29_loss0.000048.pth',
        'config': '/home/dpietsch/4DCT/configs/LearnableFilter4DLUT.yml',
    },
    'LearnableFilterSPT4DLUT': {
        'parameters': '/home/dpietsch/4DCT/parameters/LearnableFilterSPT4DLUT_epoch29_loss0.000042.pth',
        'config': '/home/dpietsch/4DCT/configs/LearnableFilterSPT4DLUT.yml',
    }

}

#------------------------------------------------------------------------------------
# loading
#------------------------------------------------------------------------------------

def build(ckpt_path, config_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError('checkpoint is not valid')

    config = ckpt.get('config')
    if config is None:
        with open(config_path) as f:
            config = yaml.safe_load(f)

    model = FilterBank(config['model']['stages']).to(DEVICE).eval()
    model.load_state_dict(ckpt['model'])

    return model, config, ckpt

def load_models():
    models, configs, ckpts = {}, {}, {}
    for method, paths in MODELS.items():
        models[method], configs[method], ckpts[method] = build(paths['parameters'], paths['config'])
          
    return models, configs, ckpts

def initial_model(config):
    return FilterBank(config['model']['stages']).eval()

def load_scan(row, margin=MARGIN):
    clean = np.concatenate([np.load(p) for p in row['clean']], axis=0)
    noisy = np.concatenate([np.load(p) for p in row['noise']], axis=0)

    pad = ((0, 0), (0, 0), (margin, margin),  (margin, margin))
    clean, noisy = np.pad(clean, pad), np.pad(noisy, pad)

    y0, y1, x0, x1 = body_box(clean, pad=margin)
    to_t = lambda a: torch.from_numpy(a[..., y0:y1, x0:x1].astype(np.float32))[None, None].to(DEVICE)
    return to_t(clean), to_t(noisy), (y0, x0)

def load_prediction(row, method):
    return torch.from_numpy(np.stack([np.load(p) for p in row[method]]))[None, None].to(DEVICE)

def roi_in_frame(scan_id, origin, shape):
    d0, d1, r0, r1, c0, c1 = AORTA_ROI[scan_id]
    oy, ox = origin
    r0, r1, c0, c1 = r0 - oy, r1 - oy, c0 - ox, c1 - ox
    H, W = shape[-2:]
    if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
        raise ValueError(f'{scan_id}: ROI lies outside the evaluated frame')
    return d0, d1, r0, r1, c0, c1

#------------------------------------------------------------------------------------
# inference
#------------------------------------------------------------------------------------

@torch.no_grad()
def denoise(noisy, model, chunk=64, overlap=MARGIN):
    D = noisy.shape[3]
    if chunk <= 0 or D <= chunk:
        return model(noisy)

    out = torch.empty_like(noisy)
    start = 0
    while start < D:
        lo, hi = max(start - overlap, 0), min(start + chunk + overlap, D)
        block = model(noisy[:, :, :, lo:hi])
        keep_lo = start - lo
        keep_n = min(chunk, D - start)
        out[:, :, :, start:start + keep_n] = block[:, :, :, keep_lo:keep_lo + keep_n]
        start += chunk
    return out

#------------------------------------------------------------------------------------
# kernels and luts
#------------------------------------------------------------------------------------

SPACING = np.array([1.0, 1.16, 1.16, 2.5])
AXES = ['t (phases)', 'row (mm)', 'col (mm)', 'z (mm)']
INK, REFERENCE = '#1f4e79', '#8c8c8c'


def offsets(shape):
    return [(np.arange(n) - n // 2) * s for n, s in zip(shape, SPACING)]

def marginals(k):
    return [k.sum(axis=tuple(a for a in range(4) if a != axis)) for axis in range(4)]

def moments(k):
    """Effective sigma per axis and the 4x4 correlation matrix, in phases and mm."""
    grids = np.meshgrid(*offsets(k.shape), indexing='ij')
    coords = np.stack([g.ravel() for g in grids])
    w = k.ravel()
    centred = coords - (coords @ w)[:, None]
    cov = (centred * w) @ centred.T
    sd = np.sqrt(np.diag(cov))
    return sd, cov / np.outer(sd, sd)

def separability(k):
    """Share of the kernel's energy explained by best product f(t) * g(space)"""
    s = np.linalg.svd(k.reshape(k.shape[0], -1), compute_uv=False)
    return (s[0] ** 2) / (s ** 2).sum()

def edge_mass(k):
    inner = k[tuple(slice(1, -1) if n > 1 else slice(None) for n in k.shape)].sum()
    return 1.0 - inner

def time_distance_map(k):
    """Mean weight per (temporal offset, spatial distance in mm)"""
    t_off, *space = offsets(k.shape)
    r = np.sqrt(sum(g ** 2 for g in np.meshgrid(*space, indexing='ij'))).round(3).ravel()
    radii = np.unique(r)
    table = np.array([[k[t].ravel()[r == rad].mean() for rad in radii] for t in range(k.shape[0])])
    return t_off, radii, table

def spread_per_phase(k):
    """RMS spatial extend (mm) along row, col, z at each temporal offset."""
    t_off, *space = offsets(k.shape)
    grids = np.meshgrid(*space, indexing='ij')
    rows = [[np.sqrt(((k[t] / k[t].sum()) * g ** 2).sum()) for g in grids] for t in range(k.shape[0])]
    return t_off, np.array(rows)

def plot_stage(name, learned, initial, out_prefix):
    # marginals against Gaussian start
    fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=False)
    for ax, x, m_l, m_i, label in zip(axes, offsets(learned.shape), marginals(learned), marginals(initial), AXES):
        ax.plot(x, m_i, color=REFERENCE, lw=2, ls='--', label='initial Gaussian')
        ax.plot(x, m_l, color=INK, lw=2, marker='o', ms=5, label='learned')
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('weight share')
    axes[0].legend(frameon=False)
    fig.suptitle(f'{name}: marginal profiles')
    fig.tight_layout()
    fig.savefig(f'{out_prefix}_marginals.png', dpi=200)
    plt.close(fig)

    if learned.shape[1:] == (1, 1, 1):
        return

    # temporal offset x spatial distance, learned and difference to start
    t_off, radii, m_l = time_distance_map(learned)
    _, _, m_i = time_distance_map(initial)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    im = axes[0].pcolormesh(radii, t_off, m_l, cmap='Blues', shading='nearest')
    fig.colorbar(im, ax=axes[0], label='mean weight')
    diff = m_l - m_i
    lim = max(np.abs(diff).max(), 1e-12)
    im = axes[1].pcolormesh(radii, t_off, diff, cmap='coolwarm', shading='nearest',
                            norm=TwoSlopeNorm(0.0, -lim, lim))
    fig.colorbar(im, ax=axes[1], label='learned - initial')
    for ax, title in zip(axes, ['learned', 'change from initial Gaussian']):
        ax.set_xlabel('spatial distance (mm)')
        ax.set_ylabel('temporal offset (phases)')
        ax.set_title(title)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(f'{out_prefix}_time_distance.png', dpi=200)
    plt.close(fig)

    # spatial spread per temporal offset, flat means sperable
    t_off, spread = spread_per_phase(learned)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    for ax, col, label in zip(axes, range(3), AXES[1:]):
        _, ref = spread_per_phase(initial)
        ax.plot(t_off, ref[:, col], color=REFERENCE, lw=2, ls='--', label='initial Gaussian')
        ax.plot(t_off, spread[:, col], color=INK, lw=2, marker='o', ms=5, label='learned')
        ax.set_xlabel('temporal offset (phases)')
        ax.set_title(f'RMS extend along {label}')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('mm')
    axes[0].legend(frameon=False)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(f'{out_prefix}_spread.png', dpi=200)
    plt.close(fig)



REFERENCE_HU = [('lung', -850.0), ('fat', -100.0), ('soft tissue / blood', 40.0), ('bone', 600.0)]

def plot_lut(name, hu, learned, initial, out_prefix):
    # the learned LUT and its change from the Gaussian start
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    extent = [hu[0], hu[-1], hu[0], hu[-1]]
    im = axes[0].imshow(learned, origin='lower', extent=extent, cmap='Blues', vmin=0.0)
    fig.colorbar(im, ax=axes[0], label='weight F(centre, neighbour)')
    diff = learned - initial
    lim = max(np.abs(diff).max(), 1e-12)
    im = axes[1].imshow(diff, origin='lower', extent=extent, cmap='coolwarm',
                        norm=TwoSlopeNorm(0.0, -lim, lim))
    fig.colorbar(im, ax=axes[1], label='learned - start')
    for ax, title in zip(axes, ['learned LUT', 'change from Gaussian start']):
        ax.set_xlabel('neighbour intensity (HU)')
        ax.set_ylabel('centre intensity (HU)')
        ax.set_title(title)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(f'{out_prefix}_lut.png', dpi=200)
    plt.close(fig)

    # range profile F(a, a + delta) for fixed centre intensities (shows intensity dependence)
    fig, axes = plt.subplots(1, len(REFERENCE_HU), figsize=(4 * len(REFERENCE_HU), 3), sharey=True)
    for ax, (label, a) in zip(axes, REFERENCE_HU):
        p = int(np.abs(hu - a).argmin())
        ax.plot(hu - hu[p], initial[p], color=REFERENCE, lw=2, ls='--', label='start (Gaussian)')
        ax.plot(hu - hu[p], learned[p], color=INK, lw=2, label='learned')
        ax.set_xlim(-600, 600)
        ax.set_xlabel('neighbour - centre (HU)')
        ax.set_title(f'centre {hu[p]:.0f} HU ({label})')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('weight')
    axes[0].legend(frameon=False)
    fig.suptitle(f'{name}: range profiles')
    fig.tight_layout()
    fig.savefig(f'{out_prefix}_profiles.png', dpi=200)
    plt.close(fig)

#------------------------------------------------------------------------------------
# metrics
#------------------------------------------------------------------------------------

def to_hu(x):
    return x * HU_RANGE + HU_MIN

def lung_window(x):
    return ((to_hu(x) - LUNG_LO) / (LUNG_HI - LUNG_LO)).clamp(0.0, 1.0)

def motion_mask(clean, quantile=MOTION_QUANTILE):
    """Gives us the voxels, whose area change the most during breathing cycle. Filter, which
    buys noise reduction by averaging across motion will degrade here. KEY FINDING!"""
    time_std = clean.std(dim=2)
    flat = time_std.flatten().float()
    if flat.numel() > 1_000_000:
        flat = flat[torch.randint(0, flat.numel(), (1_000_000, ), device=flat.device)]
    return time_std >= torch.quantile(flat, quantile)

def masked_psnr(pred, target, mask, data_range=1.0):
    """PSNR in db"""
    err = (pred - target)[mask]
    if err.numel() == 0:
        return float('inf')
    return (10 * torch.log10(data_range ** 2 / (err ** 2).mean())).item()

def volume_metrics(pred, target, mask=None):
    """pred, target: [1, 1, D, H, W]"""
    out = {
        'psnr': psnr_fn(pred, target, data_range=1.0).item(),
        'rmse_hu': (((pred - target) ** 2).mean().sqrt() * HU_RANGE).item(),
        'ssim_vol': ssim_fn(pred, target, data_range=1.0).item(),
    }

    #Slice wise SSIM on the lung window
    p = lung_window(pred)[0].permute(1, 0, 2, 3)
    t = lung_window(target)[0].permute(1, 0, 2, 3)
    out['ssim_lung'] = ssim_fn(p, t, data_range=1.0).item()
    out['psnr_lung'] = psnr_fn(p, t, data_range=1.0).item()

    if mask is not None:
        out['psnr_motion'] = masked_psnr(pred, target, mask)
        out['psnr_static'] = masked_psnr(pred, target, ~mask)

    return out

#------------------------------------------------------------------------------------
# stages
#------------------------------------------------------------------------------------

def stage_metrics(models, df, out_csv):
    rows = []
    for scan_id, row in tqdm(list(df.iterrows()), desc='metrics'):
        clean, noisy, _ = load_scan(row)

        volumes = {'noisy': interior(noisy, MARGIN)}
        for method, model in models.items():
            volumes[method] = interior(load_prediction(row, method), MARGIN)

        clean = interior(clean, MARGIN)
        mask = motion_mask(clean)
        for method, vol in volumes.items():
            for phase in range(clean.shape[2]):
                m = volume_metrics(vol[:, :, phase], clean[:, :, phase], mask)
                rows.append(dict(scan=scan_id, method=method, phase=phase, **m))

        del clean, noisy, volumes, mask
        torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'wrote {out_csv}')

def stage_roi(models, df, out_csv):
    """Std inside a homogeneous aorta box, in HU"""

    rows = []
    for scan_id, row in tqdm(list(df.iterrows()), desc='roi'):
        if scan_id not in AORTA_ROI:
            print(f'no aorta ROI for scan {scan_id}, skipping')
            continue
        d0, d1, r0, r1, c0, c1 = AORTA_ROI[scan_id]
        clean, noisy, origin = load_scan(row)

        volumes = {'reference': inplane_interior(clean, MARGIN),
                   'noisy': inplane_interior(noisy, MARGIN)}
        for method, model in models.items():
            volumes[method] = inplane_interior(load_prediction(row, method), MARGIN)

        d0, d1, r0, r1, c0, c1 = roi_in_frame(scan_id, origin, volumes['reference'].shape)

        ref_std = None
        for method, vol in volumes.items():
            roi = to_hu(vol[0, 0, PHASE_INDEX, d0:d1, r0:r1, c0:c1])
            std = roi.std().item()
            if method == 'reference':
                ref_std = std
            rows.append(dict(scan=scan_id, method=method,
                             mean_hu=roi.mean().item(), std_hu=std,
                             std_ratio_to_reference=std / ref_std if ref_std else float('nan')))

        del clean, noisy, volumes
        torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'wrote {out_csv}')

def stage_params(models, configs, out_csv):
    """Overview over learned parameters"""
    rows = []
    for method, model in models.items():
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        scalars = model.scalar_parameters()
        rows.append(dict(method=method, trainable_parameters=total,
                          stages=len(model.stages), **scalars))

        for name, param in model.named_parameters():
            if param.numel() > 1:
                path = RESULTS / f'param_{method}_{name.replace(".", "_")}.npy'
                np.save(path, param.detach().cpu().numpy())

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'wrote {out_csv}')

def stage_stats(metrics_csv, out_csv):
    from scipy.stats import wilcoxon

    df = pd.read_csv(metrics_csv)
    df = df[df['phase'] == PHASE_INDEX]
    methods = [m for m in df['method'].unique() if m != 'noisy']

    rows = []
    for metric in ['psnr', 'psnr_lung', 'ssim_lung', 'rmse_hu', 'psnr_motion']:
        table = df.pivot(index='scan', columns='method', values=metric)
        for i, a in enumerate(methods):
            for b in methods[i + 1:]:
                paired = table[[a, b]].dropna()
                if len(paired) < 3:
                    continue

                stat, p = wilcoxon(paired[a], paired[b])
                rows.append(dict(metric=metric, method_a=a, method_b=b,
                                 n=len(paired),
                                 median_a=paired[a].median(),
                                 median_b=paired[b].median(),
                                 median_difference=(paired[a] - paired[b]).median(),
                                 statistics=stat, p_value=p))

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'wrote {out_csv}')

def stage_figures(models, df, metrics_csv):
    metrics = pd.read_csv(metrics_csv)
    order = ['noisy'] + [m for m in MODELS if m in set(metrics['method'])]

    """Paper metrics"""
    at_phase = metrics[metrics['phase'] == PHASE_INDEX]
    for metric, label in [('psnr', 'PSNR (dB)'), ('psnr_lung', 'PSNR (lung window)'), ('ssim_lung', 'SSIM (lung window)'),
                          ('rmse_hu', 'RMSE (HU)')]:
        data = [at_phase[at_phase['method'] == m][metric].values for m in order]
        fig, ax = plt.subplots(figsize=(1.4 * len(order) + 2, 4))
        ax.boxplot(data, tick_labels=order)
        ax.set_ylabel(label)
        ax.tick_params(axis='x', rotation=30)
        fig.tight_layout()
        fig.savefig(RESULTS / f'box_{metric}.png', dpi=200)
        plt.close(fig)

    """maximum motion phases"""
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in order:
        per_phase = metrics[metrics['method'] == method].groupby('phase')['psnr'].mean()
        ax.plot(per_phase.index, per_phase.values, marker='o', label=method)

    ax.set_xlabel('breathing phase index')
    ax.set_ylabel('PSNR (db), mean over scans')
    ax.legend(fontsize='small')
    fig.tight_layout()
    fig.savefig(RESULTS / 'per_phase_psnr.png', dpi=200)
    plt.close(fig)

    """Motion masked against static PSNR, per method"""
    fig, ax = plt.subplots(figsize=(7, 4))
    summary = at_phase.groupby('method')[['psnr_motion', 'psnr_static']].mean().reindex(order)
    x = np.arange(len(summary))
    ax.bar(x - 0.2, summary['psnr_static'], 0.4, label='static voxels')
    ax.bar(x + 0.2, summary['psnr_motion'], 0.4, label='moving voxels')
    ax.set_xticks(x, summary.index, rotation=30)
    ax.set_ylabel('PSNR (db)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / 'motion_vs_static.png', dpi=200)
    plt.close(fig)

    stage_qualitative(models, df)

def stage_qualitative(models, df):
    for scan_id, row in tqdm(list(df.iterrows()), desc='qualitative'):
        clean, noisy, origin = load_scan(row)
        volumes = {'reference': inplane_interior(clean, MARGIN),
                   'noisy': inplane_interior(noisy, MARGIN)}
        for method, model in models.items():
            volumes[method] = inplane_interior(load_prediction(row, method), MARGIN)

        roi = roi_in_frame(scan_id, origin, volumes['reference'].shape) \
            if scan_id in AORTA_ROI else None

        d = (roi[0] + roi[1]) // 2 if roi is not None else clean.shape[3] // 2
        panels = []
        for method, vol in volumes.items():
            img = lung_window(vol[0, 0, PHASE_INDEX, d]).cpu().numpy()
            img = cvtColor((img * 255).astype(np.uint8), COLOR_GRAY2RGB)
            if roi is not None:
                _, _, r0, r1, c0, c1 = roi
                rectangle(img, (c0, r0), (c1, r1), (255, 165, 0), 1)
            panels.append(img)
            Image.fromarray(img).save(RESULTS / f'{scan_id}_{method}.png')

        Image.fromarray(np.concatenate(panels, axis=1)).save(
            RESULTS / f'{scan_id}_panel.png')

        del clean, noisy, volumes
        torch.cuda.empty_cache()

@torch.no_grad()
def stage_luts(models, configs, out_dir=RESULTS / 'luts'):
    out_dir.mkdir(parents=True, exist_ok=True)
    for method, model in models.items():
        start = initial_model(configs[method])
        for i, (stage, stage0) in enumerate(zip(model.stages, start.stages)):
            if not hasattr(stage, 'activated_lut'):
                continue
            learned = stage.activated_lut().double().cpu().numpy()        # normalised to mean diagonal 1
            initial = stage0.activated_lut().double().cpu().numpy()
            hu = np.linspace(0.0, stage.lut_max, stage.lut_bins) * HU_RANGE + HU_MIN
            name = f'{method}_stage{i + 1}'
            np.save(out_dir / f'{name}_lut.npy', learned)
            plot_lut(name, hu, learned, initial, out_dir / name)
            print(f'{name}: LUT {stage.lut_bins} bins up to {hu[-1]:.0f} HU, '
                  f'max |change| {np.abs(learned - initial).max():.3f}')


@torch.no_grad()
def stage_kernels(models, configs, out_dir=RESULTS / 'kernels'):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, model in models.items():
        start = initial_model(configs[method])
        for i, (stage, stage0) in enumerate(zip(model.stages, start.stages)):
            if not hasattr(stage, 'kernel_weights'):
                continue
            learned = stage.kernel_weights().double().cpu().numpy()
            initial = stage0.kernel_weights().double().cpu().numpy()
            name = f'{method}_stage{i + 1}'
            with np.errstate(invalid='ignore', divide='ignore'):
                sd_l, corr_l = moments(learned)
                sd_i, _ = moments(initial)
            rows.append(dict(model=method, stage=i + 1,
                             **{f'sigma_{a.split()[0]}_start': s for a, s in zip(AXES, sd_i)},
                             **{f'sigma_{a.split()[0]}_learned': s for a, s in zip(AXES, sd_l)},
                             separability_start=separability(initial), separability_learned=separability(learned),
                             edge_mass_start=edge_mass(initial), edge_mass_learned=edge_mass(learned)))
            plot_stage(name, learned, initial, out_dir / name)
    
    pd.DataFrame(rows).to_csv(out_dir / 'kernel_summary.csv', index=False)
    print(f'wrote {out_dir}/kernel_summary.csv')

def prediction_files(method, scan_id, n_phases):
    result_dir = DENOISE_DIR / method / scan_id
    return [result_dir / f'phase_0{i}.npy' for i in range(n_phases)]

def predictions_current(method):
    marker = DENOISE_DIR / method / 'checkpoint.txt'
    return marker.exists() and marker.read_text().strip() == str(MODELS[method]['parameters'])

@torch.no_grad()
def ensure_predictions(models, df, force=False):
    todo = {}
    for method in models:
        stale = force or not predictions_current(method)
        missing = [scan_id for scan_id, row in df.iterrows()
                   if stale or not all(p.exists() for p in prediction_files(method, scan_id, len(row['noise'])))]

        if missing:
            todo[method] = set(missing)
            reason = 'forced' if force else ('new checkpoint' if stale else 'missing files')
            print(f'{method}: denoising {len(missing)} scan(s) ({reason})')

    scans = [scan_id for scan_id in df.index if any(scan_id in s for s in todo.values())]
    for scan_id in tqdm(scans, desc='denoise'):
        row = df.loc[scan_id]
        _, noisy, _ = load_scan(row)
        for method, model in models.items():
            if scan_id not in todo.get(method, ()):
                continue
            denoised = denoise(noisy, model).cpu().numpy()
            files = prediction_files(method, scan_id, len(row['noise']))
            files[0].parent.mkdir(parents=True, exist_ok=True)
            for i, path in enumerate(files):
                np.save(path, denoised[0, 0, i].astype(np.float32))
            del denoised
        del noisy
        torch.cuda.empty_cache()

    for method in todo:
        (DENOISE_DIR / method / 'checkpoint.txt').write_text(str(MODELS[method]['parameters']))

    for method in models:
        df[method] = pd.Series({scan_id: [str(p) for p in prediction_files(method, scan_id, len(row['noise']))]
                                for scan_id, row in df.iterrows()})
    return df


def summarize(metrics_csv):
    df = pd.read_csv(metrics_csv)
    at_phase = df[df['phase'] == PHASE_INDEX]
    at_phase = at_phase.assign(batch=np.where(at_phase['scan'].isin([f'case_{i:02d}' for i in range(1,6)]), '01-05', '06-10'))

    cols = ['psnr', 'psnr_lung', 'ssim_lung', 'ssim_vol', 'rmse_hu', 'psnr_motion', 'psnr_static']
    print(f'\nmean +- std over scans at phase {PHASE_INDEX}\n')
    print(at_phase.groupby(['batch', 'method'])[cols].agg(['mean', 'std']).round(4).to_string())

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', nargs='+',
                        default=['metrics', 'roi', 'params', 'figures', 'stats', 'kernels', 'luts'],
                        choices=['metrics', 'roi', 'params', 'figures', 'stats', 'kernels', 'luts', 'denoise'])

    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True, parents=True)
    metrics_csv = RESULTS / 'test_metrics.csv'
    df = CTValidationDataset.generate_dataframe(TEST_DATA)
    models, configs, ckpts = load_models()
    needs_predictions = {'metrics', 'roi', 'figures', 'denoise'} & set(args.stage)
    if needs_predictions:
        df = ensure_predictions(models, df, force='denoise' in args.stage)

 
    if 'metrics' in args.stage:
        stage_metrics(models, df, metrics_csv)
    if 'roi' in args.stage:
        stage_roi(models, df, RESULTS / 'roi_std.csv')
    if 'params' in args.stage:
        stage_params(models, configs, RESULTS / 'parameters.csv')
    if 'figures' in args.stage:
        stage_figures(models, df, metrics_csv)
    if 'stats' in args.stage:
        stage_stats(metrics_csv, RESULTS / 'paired_tests.csv')
    if 'kernels' in args.stage:
        stage_kernels(models, configs)
    if 'luts' in args.stage:
        stage_luts(models, configs)

    if metrics_csv.exists():
        summarize(metrics_csv)


if __name__ == '__main__':
    main()