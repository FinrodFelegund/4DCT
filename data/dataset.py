from torch.utils.data import Dataset
import pandas as pd
from typing import List
import torch
import os
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    SpatialPadd,
    ConcatItemsd,
    DeleteItemsd,
    EnsureTyped,
    Orientationd,
    CropForegroundd,
)

from .CustomTransforms import (
    OrderChannels,
    EnsureChannelDimension,
    EnsureBatchDimension,
    CacheDataSet,
    CropBatch,
    GenerateNoiseScans,
)

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
        clean = sample['clean']
        scan = sample['scan']

        clean_phase_keys = [f'phase_0{i}' for i in range(len(clean))]
        phase_dict = {}
        for key_clean, volume_clean in zip(clean_phase_keys, clean):
            phase_dict[key_clean] = volume_clean
        phase_dict['scan'] = scan

        tensor4d = self.transforms(phase_dict)

        return tensor4d


    @staticmethod
    def generate_dataframe(data: List[str] | str, handle: str = 'phase_**.nii**'):
        """Generate a DataFrame, which contains all the paths to the phase series files of a scan."""
        
        if type(data) == str:
            data = [data]
        
        data_dict = {}
        for dataset in data:
            scans = sorted(os.listdir(dataset))
            for scan in scans:
                scan_dir = os.path.join(dataset, scan)
                clean = sorted(glob(f'{scan_dir}/images/phase_**.nii'))
                data_dict[scan] = {
                    'clean': clean,
                    'scan': scan,
                }
        
        
        return pd.DataFrame(data_dict).transpose()
    
    @staticmethod
    def collate_fn(data):
        gt_data = [tensor['clean'].as_tensor() for tensor in data]
        gt_data = torch.cat(gt_data, dim=0)
        noise_data = [tensor['noise'].as_tensor() for tensor in data]
        noise_data = torch.cat(noise_data, dim=0)
        return gt_data, noise_data
    
    @staticmethod
    def cache_collate(data):
        out = {}
        for sample in data:
            img_shapes = []

            keys = list(sample.keys())
            for key in keys:
                if not key == 'scan':
                    img_shapes.append(sample[key].shape)
            scan = sample['scan']
            out[scan] = np.array(img_shapes)
            
        return out

    
    @staticmethod
    def get_train_transforms():
        clean_keys = [f'phase_0{i}' for i in range(10)]
        noise_keys = [f'noisy_phase_0{i}' for i in range(10)]
        transforms = Compose([
            LoadImaged(keys=clean_keys + noise_keys),
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
            EnsureChannelDimension(keys=['clean', 'noise']),
            EnsureTyped(keys=['clean', 'noise']),
            CropBatch(keys=clean_keys + noise_keys, num_crops=4)
        ])

        return transforms
    
    @staticmethod
    def get_validation_transforms():
        clean_keys = [f'phase_0{i}' for i in range(10)]
        noise_keys = [f'noisy_phase_0{i}' for i in range(10)]
        transforms = Compose([
            LoadImaged(keys=clean_keys + noise_keys),
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
            EnsureChannelDimension(keys=['clean', 'noise']),
            EnsureBatchDimension(keys=['clean', 'noise']),
            EnsureTyped(keys=['clean', 'noise']),
        ])

        return transforms
    
    @staticmethod
    def get_cache_transforms(destination: str):
        clean_keys = [f'phase_0{i}' for i in range(10)]
        noise_keys = [f'noisy_phase_0{i}' for i in range(10)]
        transforms = Compose([
            LoadImaged(keys=clean_keys),
            EnsureChannelFirstd(keys=clean_keys),
            Orientationd(keys=clean_keys, axcodes='RAS'),
            Spacingd(
                keys=clean_keys,
                pixdim=[1.16, 1.16, 2.5],
                mode=['bilinear'] * len(clean_keys),
            ),
            CropForegroundd(keys=clean_keys, select_fn=lambda x: x > -800, source_key='image'),
            SpatialPadd(
                keys=clean_keys,
                spatial_size=[32, 32, 32],
                mode='constant',
                constant_values=0,
            ),
            OrderChannels(keys=clean_keys),
            ScaleIntensityRanged(
                keys=clean_keys,
                a_min=--1024,
                a_max=3000,
                b_min=-1024,
                b_max=3000,
                clip=True,
            ),
            GenerateNoiseScans(keys=clean_keys),
            CacheDataSet(keys=clean_keys + noise_keys, destination=destination)
        ])

        return transforms
    
    
def cacheDataSet():
    DATASETS = ['/home/dpietsch/Pictures/Inhouse1', '/home/dpietsch/Pictures/Inhouse2']
    cache_dest = '/home/dpietsch/Pictures/.cache'
    Path(cache_dest).mkdir(parents=True, exist_ok=True)
    DESTINATIONS = [f'{cache_dest}/Inhouse1Processed', f'{cache_dest}/Inhouse2Processed']


    for dataset, destination in zip(DATASETS, DESTINATIONS):
        dataframe = CT4dDataset.generate_dataframe(dataset)
        cache_transforms = CT4dDataset.get_cache_transforms(destination=destination)
        dataset = CT4dDataset(dataframe=dataframe, transforms=cache_transforms)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, shuffle=False, num_workers=0, collate_fn=CT4dDataset.cache_collate)
        tqdm_loader = tqdm(dataloader)
        for index, out in enumerate(tqdm_loader):
            keys = list(out.keys())
            shapes = [out[key].shape for key in keys]
            tqdm_loader.set_description(f'Cached batch with index {index}, scans: {keys}, shapes: {shapes}')


