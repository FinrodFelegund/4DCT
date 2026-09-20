import torch
import pandas as pd
import numpy as np
from models.SPBF import FilterBank
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_fn
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
from data.dataset import CTValidationDataset
from tqdm import tqdm
from PIL import Image
import cv2

DEVICE = 'cuda'

def scan_metrics(target: torch.Tensor, preds: torch.Tensor):

    psnr = psnr_fn(preds=preds, target=target, data_range=(0.0, 1.0)).item()
    ssim = ssim_fn(preds=preds, target=target, data_range=(0.0, 1.0)).item()
    return psnr, ssim

def compute_metrics(csv_path: str = 'results/test_metrics.csv'):
    df = CTValidationDataset.generate_dataframe('/home/dpietsch/Data/.cache/Inhouse2Processed')
    models = {
        'spatial': build(3, 0, 0, '/home/dpietsch/4DCT/parameters/SPBF_Filterbank_Baseline_loss0.000034.pth'),
        'joint': build(3, 1, 0, '/home/dpietsch/4DCT/parameters/SPBF_Filterbank_Joint_loss0.000028.pth'),
        'spatio_temporal': build(0, 0, 3, '/home/dpietsch/4DCT/parameters/SPBF_Filterbank_SpatialTemporal_loss0.000029.pth'),
    }



    rows = []
    tqdm_df = tqdm(df.iterrows())
    for scan_id, row in tqdm_df:
        clean, noisy = load_scan(row)
        psnr, ssim = scan_metrics(target=clean[:, :, 4], preds=noisy[:, :, 4])
        rows.append(dict(scan=scan_id, method='Noisy', psnr=psnr, ssim=ssim))
        for method, model in models.items():
            prediction = denoise(noisy, model)
            #we evaluate on the 50% inhale phase
            psnr, ssim = scan_metrics(target=clean[:, :, 4], preds=prediction[:, :, 4])
            rows.append(dict(scan=scan_id, method=method, psnr=psnr, ssim=ssim))
            tqdm_df.set_postfix_str(f'Scan: {scan_id} | Model: {method}')

    res = pd.DataFrame(rows)
    res.to_csv(csv_path, index=False)

def build_figures(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    df = df.groupby(by=['method'])

def build(num_spatial, num_temporal, num_spt, ckpt):
    m = FilterBank(num_spatial=num_spatial, num_temporal=num_temporal, num_spt=num_spt).to(DEVICE).eval()
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return m


def load_scan(row):
    clean = np.concatenate([np.load(p) for p in row['clean']], axis=0)
    noisy = np.concatenate([np.load(p) for p in row['noise']], axis=0)
    T, D, H, W = clean.shape
    center_y, center_x = H // 2, W // 2
    clean_cropped, noisy_cropped = clean[:, :, center_y-110:center_y+110, center_x-120:center_x+120], noisy[:, :, center_y-110:center_y+110, center_x-120:center_x+120]

    to_t = lambda a: torch.from_numpy(a.astype(np.float32))[None, None]
    
    return to_t(clean_cropped).to(DEVICE), to_t(noisy_cropped).to(DEVICE)

@torch.no_grad()
def denoise(noisy: torch.Tensor, model: torch.nn.Module):
    prediction = model(noisy)
    return prediction
    
def quantitative(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    grouped = df.groupby('method')[['psnr', 'ssim']].agg(list).to_dict()
    for key, value in grouped.items():
        fig, ax = plt.subplots()
        ax.set_ylabel(key)
        results = list(value.values())
        models = list(value.keys())
        bplot = ax.boxplot(results,
                           tick_labels=models)
        fig.tight_layout()
        plt.savefig(f'results/{key}.png')

def roi_std(vol_norm, r0, r1, c0, c1, hu_min, hu_range):
    """vol_norm: [H, W] normalized slice. Returns (mean, std, min, max) in HU."""
    roi = vol_norm[r0:r1, c0:c1] * hu_range + hu_min
    return roi.mean(), roi.std(), roi.min(), roi.max()

def qualitative(hu_range=(-500, 1500)):
    df = CTValidationDataset.generate_dataframe('/home/dpietsch/Data/.cache/Inhouse2Processed')
    models = {
        'spatial': build(3, 0, 0, '/home/dpietsch/4DCT/parameters/SPBF_Filterbank_Baseline_loss0.000034.pth'),
        'joint': build(3, 1, 0, '/home/dpietsch/4DCT/parameters/SPBF_Filterbank_Joint_loss0.000028.pth'),
        'spatio_temporal': build(0, 0, 3, '/home/dpietsch/4DCT/parameters/SPBF_Filterbank_SpatialTemporal_loss0.000029.pth'),
    }

    results = []
    tqdm_df = tqdm(df.iterrows())
    for scan_id, row in tqdm_df:
        d = {} 
        clean, noisy = load_scan(row)
        d['clean'] = clean
        d['noisy'] = noisy
        for method, model in models.items():
            prediction = denoise(noisy, model)
            d[method] = prediction
        results.append(d)
        

    HU_MIN, HU_MAX = -1024.0, 3000.0
    HU_RANGE = HU_MAX - HU_MIN

    def to_hu(x):
        return x * HU_RANGE + HU_MIN


    def apply_window(x_norm, lo_hu, hi_hu):
        hu = to_hu(x_norm)
        return ((hu - lo_hu) / (hi_hu - lo_hu)).clip(0.0, 1.0)

    X0, Y0, W, H = 53, 65, 20, 20      # top-left in image coords
    R0, R1, C0, C1 = Y0, Y0 + H, X0, X0 + W

    for res in results:
        for method, scan in res.items():
            scan = scan.cpu().numpy()[0, 0, 4, 60]
            disp = apply_window(scan, -500, 1500)
            disp = (disp * 255.).astype(np.uint8)
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
            cv2.rectangle(disp, (X0, Y0), (X0 + W, Y0 + H), (255, 165, 0), 1)
            Image.fromarray(disp[R0-30:R1+30, C0-30:C1+30]).save(f'results/{method}.png')
            mean, std, lo, hi = roi_std(scan, R0, R1, C0, C1, HU_MIN, HU_RANGE)
            print(f'{method:16s} mean={mean:7.1f}  std={std:6.1f}  range=[{lo:.0f}, {hi:.0f}]')


#94, 108
#orange (255, 165, 0)

def calculate_mean_std(df: pd.DataFrame):
    grouped = df.groupby('method')[['psnr', 'ssim']].agg(list).to_dict()

    for key, value in grouped.items():
        print(key)
        for sub_key, sub_value in value.items():
            print(sub_key)
            print(f'{np.array(sub_value).mean():.4f}', f'{np.array(sub_value).std():.4f}')

if __name__ == '__main__':
    qualitative()



