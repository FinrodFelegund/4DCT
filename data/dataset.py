from torch.utils.data import Dataset
import pandas as pd
from typing import List
from typing import Dict
import glob
import numpy as np
import torch
import os
from pathlib import Path
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    SpatialPadd,
    RandSpatialCropd,
    CenterSpatialCropd,
    ConcatItemsd,
    DeleteItemsd,
    EnsureTyped
)
from monai.transforms import MapTransform
from utils import noise

class NoiseTransform(MapTransform):
    def __init__(self, keys, allow_missing_keys = False, dosage: float = 0.5):
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.dosage = dosage

    def __call__(self, data: Dict):
        key = list(data)[0]
        _, H, _, _ = data[key].shape
        noisepipeline = noise.SinogramNoise(b_type='Fan-Beam', image_size=H, dosage=self.dosage)
        keys = list(data.keys())
        for key in keys:
            try:
                with torch.no_grad():
                    data[f'{key}_noise'] = noisepipeline(data[key])
            except Exception as e:
                raise RuntimeError(str(e))
        return data

class CT4dDataset(Dataset):

    def __init__(
            self,
            dataframe: pd.DataFrame,
            transforms: List[None]
              ):
        
        self.dataframe = dataframe
        self.len = len(self.dataframe)
        self.transforms = transforms


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        volumes = sorted(glob.glob(f'{sample['phase_paths']}/phase_*.nii*'))
        phase_keys = [f'phase_0{i}' for i in range(len(volumes))]
        phase_dict = {}
        for key, volume in zip(phase_keys, volumes):
            phase_dict[key] = volume 

        data_dicts = [phase_dict]
        tensor4d = self.transforms(data_dicts)

        return tensor4d


    @staticmethod
    def generate_dataframe(data: List[str] | str, handle: str = 'phase_**.nii**'):
        """Generate a pickle file, which contains all the paths to the phase series files of a scan."""
        
        if type(data) == str:
            data = [data]
        
        data_dict = {}
        for dataset in data:
            for root, dirs, files in os.walk(dataset):
                for file in files:
                    if 'phase' in file:
                        data_dict[root] = file
        
        return pd.DataFrame(data=list(data_dict.keys()), columns=['phase_paths'])
    
    @staticmethod
    def collate_fn(data: List[dict]):
        gt_data = [tensor[0]['clean'].as_tensor() for tensor in data]
        gt_data = torch.stack(gt_data, dim=0).unsqueeze(1)
        noise_data = [tensor[0]['noise'].as_tensor() for tensor in data]
        noise_data = torch.stack(noise_data, dim=0).unsqueeze(1)
        return gt_data, noise_data
    
    @staticmethod
    def get_train_transforms(dosage: float = 0.5):
        clean_keys = [f'phase_0{i}' for i in range(10)]
        noise_keys = [f'phase_0{i}_noise' for i in range(10)]
        transforms = Compose([
            LoadImaged(keys=clean_keys),
            EnsureChannelFirstd(keys=clean_keys),
            ScaleIntensityRanged(
                keys=clean_keys,
                a_min=-1000,
                a_max=1000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Spacingd(
                keys=clean_keys,
                pixdim=[1.16, 1.16, 2.5],
                mode=['bilinear'] * len(clean_keys),
            ),
            SpatialPadd(
                keys=clean_keys,
                spatial_size=[32, 32, 32],
                mode='constant',
                constant_values=0,
            ),
            NoiseTransform(keys=clean_keys, dosage=dosage),
            RandSpatialCropd(
                keys=clean_keys + noise_keys,
                roi_size=[32, 32, 32],
            ),
            ConcatItemsd(
                keys=clean_keys,
                name='clean',
                dim=0,
            ),
            ConcatItemsd(
                keys=noise_keys,
                name='noise',
                dim=0,
            ),
            DeleteItemsd(keys=clean_keys + noise_keys),
            EnsureTyped(keys=['clean', 'noise']),

        ])

        return transforms
    
    @staticmethod
    def get_validation_transforms(dosage: float = 0.5):
        clean_keys = [f'phase_0{i}' for i in range(10)]
        noise_keys = [f'phase_0{i}_noise' for i in range(10)]
        transforms = Compose([
            LoadImaged(keys=clean_keys),
            EnsureChannelFirstd(keys=clean_keys),
            ScaleIntensityRanged(
                keys=clean_keys,
                a_min=-1000,
                a_max=1000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Spacingd(
                keys=clean_keys,
                pixdim=[1.16, 1.16, 2.5],
                mode=['bilinear'] * len(clean_keys),
            ),
            SpatialPadd(
                keys=clean_keys,
                spatial_size=[32, 32, 32],
                mode='constant',
                constant_values=0,
            ),
            NoiseTransform(keys=clean_keys, dosage=dosage),
            CenterSpatialCropd(
                keys=clean_keys + noise_keys,
                roi_size=[32, 32, 32],
            ),
            ConcatItemsd(
                keys=clean_keys,
                name='clean',
                dim=0,
            ),
            ConcatItemsd(
                keys=noise_keys,
                name='noise',
                dim=0,
            ),
            DeleteItemsd(keys=clean_keys + noise_keys),
            EnsureTyped(keys=['clean', 'noise']),

        ])

        return transforms