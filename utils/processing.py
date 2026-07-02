import os
from pathlib import Path
from .noise import SinogramNoise
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt

DATASETPATH = r'/home/dpietsch/Pictures/Inhouse1'

def create_folder(name: str):
    dirs = os.listdir(DATASETPATH)

    for dir in dirs:
        Path(os.path.join(DATASETPATH, dir, name)).mkdir(parents=True, exist_ok=True)


def move_images():
    dirs = os.listdir(DATASETPATH)

    for dir in dirs:
        files = os.listdir(os.path.join(DATASETPATH, dir))
        for file in files:
            if file != 'images' and file != 'noise':
                source = os.path.join(DATASETPATH, dir, file)
                dest = os.path.join(DATASETPATH, dir, 'images', file)
                os.rename(source, dest)


DATASETPATHS = [r'/home/dpietsch/Pictures/Inhouse1', r'/home/dpietsch/Pictures/Inhouse2']

def normalize_img(img: torch.Tensor):
    img_min = img.min()
    img_max = img.max()

    img = (img - img_min) / (img_max - img_min + 1e-8)

    return img, img_min, img_max

def unnormalize(img: torch.Tensor, img_min, img_max):
    img = (img * (img_max - img_min)) + img_min
    return img

def apply_noise(img: np.ndarray):
    H, W, D = img.shape
    img = np.transpose(img, axes=(2, 0, 1))
    sNoise = SinogramNoise(b_type='Fan-Beam', image_size=H, dosage=0.25)
    img_tensor = torch.from_numpy(img).float()
    step_size = 30
    device = torch.device('cuda')

    with torch.no_grad():
        for i in range(0, len(img_tensor), step_size):
            start = i
            end = min(i + step_size, len(img_tensor))
            gpu_batch = img_tensor[start:end, :, :].to(device)
            gpu_batch, img_min, img_max = normalize_img(gpu_batch)
            gpu_batch = sNoise(gpu_batch)
            gpu_batch = unnormalize(gpu_batch, img_min, img_max)
            img_tensor[start:end, :, :] = gpu_batch.cpu()


    return np.transpose(img_tensor.numpy(), axes=(1, 2, 0))

def create_noise_dataset():
    for dataset in DATASETPATHS:
        image_paths = os.listdir(dataset)
        for scan in tqdm(image_paths):
            phase_imgs = os.listdir(os.path.join(dataset, scan, 'images'))
            for phase_img in phase_imgs:
                if 'phase' in phase_img:
                    img = os.path.join(dataset, scan, 'images', phase_img)
                    img = nib.load(img)
                    img_data = img.get_fdata()
                    img_header = img.header
                    affine = img.affine
                    noise_data = apply_noise(img_data)
                    noise_img = nib.Nifti1Image(dataobj=noise_data, affine=affine, header=img_header)
                    noise_path = os.path.join(dataset, scan, 'noise', f'noise_{phase_img}')
                    nib.save(noise_img, noise_path)

                
        

def compute_patch_offsets(D: int, H: int, W: int, patch_size: int = 32, window_step: int = 16):
    def axis_offset(d: int):
        offsets = []
        for i in range(0, d - patch_size, window_step):
            offsets.append(i)
        if i + patch_size != d:
            offsets.append(d - patch_size)
        return offsets
    
    D_offsets = axis_offset(D)
    H_offsets = axis_offset(H)
    W_offsets = axis_offset(W)
    num_patches = len(D_offsets) * len(H_offsets) * len(W_offsets)

    return D_offsets, H_offsets, W_offsets, num_patches

def make_sigma_plot(sigma_dict, title):
    rows = [
        {'type': label, 'sigma': v}
        for label, values in sigma_dict.items()
        for v in values
    ]

    if not rows:
        return None
    
    df = pd.DataFrame(rows)

    fig = px.violin(
        df, x='type', y='sigma',
        box=True,
        points='all',
        title=title
    )

    fig.update_traces(meanline_visible=True)
    means = df.groupby('type', sort=False)['sigma'].mean().reset_index()
    fig.add_scatter(
        x=means['type'], y=means['sigma'],
        mode='markers',
        marker=dict(color='red', symbol='diamond', size=11, line=dict(width=1, color='black')),
        name='mean',
    )

    fig.update_layout(
        yaxis_title='σ (activated)',
        xaxis_title=None,
        showlegend=True,
    )

    return fig