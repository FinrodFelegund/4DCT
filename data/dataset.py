from torch.utils.data import Dataset
import pandas as pd
from typing import List
import torch
import os
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
import csv
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
    ResizeWithPadOrCropd,
)

from .CustomTransforms import (
    OrderChannels,
    EnsureChannelDimension,
    CacheBatch,
    CropBatchTrain,
    CropBatchValidation,
    GenerateNoiseBatch,
    MidSlice,
    ForegroundBBox,
    AssertPhasesAligned,
    ApplyHUOffset,
)

EXCLUDED = {
    '069_4DCT_Lunge_amplitudebased_complete': 'missing slice in the middle of phase scan 6',
    '558_Lunge_amplitudebased': 'missing slice in the middle of phase scan 2',
    '529_Lunge_amplitudebased': 'coverage 66mm, 26 slices after resampling',
    '567_Lunge_amplitudebased': '4DCT motion phantom, not a patient'
}

class CTTrainDataset(Dataset):

    def __init__(
            self,
            dataframe: pd.DataFrame,
            transforms: List[None],
              ):
        
        self.dataframe = dataframe
        self.len = len(self.dataframe)
        self.transforms = transforms



    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        clean = sample['clean']
        noisy = sample['noise']
        scan = sample['scan']

        clean_phase_keys = [f'phase_0{i}' for i in range(len(clean))]
        noise_phase_keys = [f'noise_phase_0{i}' for i in range(len(noisy))]
        phase_dict = {}
        for key_clean, volume_clean, key_noisy, volume_noisy in zip(clean_phase_keys, clean, noise_phase_keys, noisy):
            phase_dict[key_clean] = volume_clean
            phase_dict[key_noisy] = volume_noisy
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
                clean = sorted(glob(f'{scan_dir}/images/phase_**.npy'))
                noisy = sorted(glob(f'{scan_dir}/noise/noise_phase_**.npy'))
                data_dict[scan] = {
                    'clean': clean,
                    'noise': noisy,
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
    def get_train_transforms(num_crops=8, patch_size=(32, 64, 64), margin=8):
        clean_keys = [f'phase_0{i}' for i in range(10)]
        noise_keys = [f'noise_phase_0{i}' for i in range(10)]
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
            ForegroundBBox(keys=['clean', 'noise'], min_size=max(patch_size[1:]) + 2 * margin),
            CropBatchTrain(keys=['clean', 'noise'], num_crops=num_crops, patch_size=patch_size, margin=margin)
        ])

        return transforms

class CTValidationDataset(Dataset):

    def __init__(
            self,
            dataframe: pd.DataFrame,
            transforms: List[None],
            scan_transforms: List[None] | None = None
              ):
        
        self.dataframe = dataframe
        self.len = len(self.dataframe)
        self.transforms = transforms
        self.scan_transforms = scan_transforms

    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        clean = sample['clean']
        noisy = sample['noise']
        scan = sample['scan']

        clean_phase_keys = [f'phase_0{i}' for i in range(len(clean))]
        noise_phase_keys = [f'noise_phase_0{i}' for i in range(len(noisy))]
        phase_dict = {}
        for key_clean, volume_clean, key_noisy, volume_noisy in zip(clean_phase_keys, clean, noise_phase_keys, noisy):
            phase_dict[key_clean] = volume_clean
            phase_dict[key_noisy] = volume_noisy
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
                clean = sorted(glob(f'{scan_dir}/images/phase_**.npy'))
                noisy = sorted(glob(f'{scan_dir}/noise/noise_phase_**.npy'))
                data_dict[scan] = {
                    'clean': clean,
                    'noise': noisy,
                    'scan': scan,
                    
                }
        
        
        return pd.DataFrame(data_dict).transpose()

    @staticmethod
    def generate_origdata_dataframe(data: List[str] | str, handle: str = 'phase_**.nii**'):
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
    def get_validation_transforms(num_crops=8, patch_size=(32, 64, 64), margin=8):
        clean_keys = [f'phase_0{i}' for i in range(10)]
        noise_keys = [f'noise_phase_0{i}' for i in range(10)]
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
            #EnsureBatchDimension(keys=['clean', 'noise']),
            EnsureTyped(keys=['clean', 'noise']),
            ForegroundBBox(keys=['clean', 'noise'], min_size=max(patch_size[1:]) + 2 * margin),
            CropBatchValidation(keys=['clean', 'noise'], 
                                num_crops_per_dim=num_crops,
                                patch_size=patch_size,
                                margin=margin)
        ])

        return transforms
    
    @staticmethod
    def get_scan_transforms():
        clean_keys = [f'phase_0{i}' for i in range(10)]
        noise_keys = [f'noise_phase_0{i}' for i in range(10)]
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
            #EnsureBatchDimension(keys=['clean', 'noise']),
            EnsureTyped(keys=['clean', 'noise']),
            MidSlice(keys=['clean', 'noise']),
        ])

        return transforms
    
    def get_whole_slice(self, idx: int = 0):
        sample = self.dataframe.iloc[idx]
        clean = sample['clean']
        noisy = sample['noise']
        scan = sample['scan']

        clean_phase_keys = [f'phase_0{i}' for i in range(len(clean))]
        noise_phase_keys = [f'noise_phase_0{i}' for i in range(len(noisy))]
        phase_dict = {}
        for key_clean, volume_clean, key_noisy, volume_noisy in zip(clean_phase_keys, clean, noise_phase_keys, noisy):
            phase_dict[key_clean] = volume_clean
            phase_dict[key_noisy] = volume_noisy
        phase_dict['scan'] = scan

        tensor4d = self.scan_transforms(phase_dict)
        return tensor4d['clean'].as_tensor(), tensor4d['noise'].as_tensor()
    

    
class CTCacheDataset(Dataset):
    def __init__(
            self,
            dataframe: pd.DataFrame,
            transforms: List[None],
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
        destination = sample['destination']

        clean_phase_keys = [f'phase_0{i}' for i in range(len(clean))]
        phase_dict = {}
        for key_clean, volume_clean in zip(clean_phase_keys, clean):
            phase_dict[key_clean] = volume_clean

        phase_dict['scan'] = scan
        phase_dict['destination'] = destination

        tensor4d = self.transforms(phase_dict)

        return tensor4d




    @staticmethod
    def generate_dataframe(data: List[str] | str, destination: List[str] | str, handle: str = 'phase_**.nii**'):
        """Generate a DataFrame, which contains all the paths to the phase series files of a scan."""
        
        if type(data) == str:
            data = [data]

        if type(destination) == str:
            destination = [destination]
        
        data_dict = {}
        for dataset, dest in zip(data, destination):
            scans = sorted(os.listdir(dataset))
            for scan in scans:
                scan_dir = os.path.join(dataset, scan)
                dest_dir = os.path.join(dest, scan)
                clean = sorted(glob(f'{scan_dir}/images/phase_**'))
                data_dict[scan] = {
                    'clean': clean,
                    'scan': scan,
                    'destination': dest_dir,
                }
        
        
        return pd.DataFrame(data_dict).transpose()

    @staticmethod
    def cache_collate(data):
        out = {}
        for idx, sample in enumerate(data):
            keys = list(sample.keys())
            for key in keys:
                    if key == 'scan' or key == 'destination':
                        out[key] = sample[key]
                    else:
                        out[f'{idx}_{key}'] = sample[key]

        return out
            
    @staticmethod
    def get_cache_transforms(dosage=0.25, i0_full=1e5, i0_by_scan=None):
        PIXEL_SIZE_IN_MM = 1.16
        HU_MIN, HU_MAX = -1024.0, 3000.0
        HU_OFFSETS = {f'case_{i:02d}': 27.0 for i in range(6, 11)}
        clean_keys = [f'phase_0{i}' for i in range(10)]#['phase_05']
        noise_keys = [f'noise_phase_0{i}' for i in range(10)]#['noise_phase_05']
        transforms = Compose([
            LoadImaged(keys=clean_keys),
            EnsureChannelFirstd(keys=clean_keys),
            ApplyHUOffset(keys=clean_keys, offsets=HU_OFFSETS),
            AssertPhasesAligned(keys=clean_keys,
                                ref_key=clean_keys[0]),
            Orientationd(keys=clean_keys, axcodes='RAS'),
            Spacingd(
                keys=clean_keys,
                pixdim=[PIXEL_SIZE_IN_MM, PIXEL_SIZE_IN_MM, 2.5],
                mode=['bilinear'] * len(clean_keys),
            ),
            ScaleIntensityRanged(
                keys=clean_keys,
                a_min=HU_MIN,
                a_max=HU_MAX,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            SpatialPadd(
                keys=clean_keys,
                spatial_size=[64, 64, 64],
                mode='constant',
                constant_values=0.0,
            ),
            # (C, H, W, D) -> (C, D, H, W)
            OrderChannels(keys=clean_keys),
            ResizeWithPadOrCropd(keys=clean_keys, spatial_size=(-1, 512, 512)),
            GenerateNoiseBatch(keys=clean_keys, dosage=dosage,
                               i0_full=i0_full, i0_by_scan=i0_by_scan,
                               pixel_size_mm=PIXEL_SIZE_IN_MM),
            CacheBatch(keys=clean_keys + noise_keys)
        ])

        return transforms

def load_i0_tables(*paths):
    table = {}
    for path in paths:
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row['i0_fit']:
                    table[row['scan']] = float(row['i0_fit'])
    return table

    
def cacheDataSet():
    DATASETS = ['/home/dpietsch/Data/Inhouse1', '/home/dpietsch/Data/Inhouse2']
    cache_dest = '/home/dpietsch/Data/.cache'
    Path(cache_dest).mkdir(parents=True, exist_ok=True)
    DESTINATIONS = [f'{cache_dest}/Inhouse1Processed_v3', f'{cache_dest}/Inhouse2Processed_v3']

    i0_by_scan = load_i0_tables('explorations/i0_train.csv', 'explorations/i0_test.csv')
    dataframe = CTCacheDataset.generate_dataframe(DATASETS, DESTINATIONS)
    dataframe = dataframe.drop(index=[s for s in EXCLUDED if s in dataframe.index])
    cache_transforms = CTCacheDataset.get_cache_transforms(dosage=0.25, i0_full=1e5, i0_by_scan=i0_by_scan)
    dataset = CTCacheDataset(dataframe=dataframe, transforms=cache_transforms)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, shuffle=False, num_workers=4, batch_size=1, collate_fn=CTCacheDataset.cache_collate)
    for i, out in enumerate((pbar := tqdm(dataloader))):
        pbar.set_description(f'{i+1}: cached {out['scan']}')

        




