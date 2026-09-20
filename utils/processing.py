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
from data.dataset import CacheDataSet

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


                
        

def compute_patch_offsets(D: int, H: int, W: int, patch_size: int = 32, window_step: int = 16):
    def axis_offset(d: int):
        offsets = []
        for i in range(0, d - patch_size, window_step):
            if i + patch_size != d:
                offsets.append(d - patch_size)
            else: 
                offsets.append(i)
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
