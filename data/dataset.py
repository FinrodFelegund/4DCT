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

)

from .CustomTransforms import (
    OrderChannels,
    EnsureChannelDimension,
    EnsureBatchDimension,
    CacheDataSet,
    CropBatchTrain,
    CropBatchValidation,
    GenerateNoiseScans,
    CustomForegroundCrop,
    PadToSquare,
    MidSlice,
)

class CT4dDataset(Dataset):

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
        noisy_phase_keys = [f'noisy_phase_0{i}' for i in range(len(noisy))]
        phase_dict = {}
        for key_clean, volume_clean, key_noisy, volume_noisy in zip(clean_phase_keys, clean, noisy_phase_keys, noisy):
            phase_dict[key_clean] = volume_clean
            phase_dict[key_noisy] = volume_noisy
        phase_dict['scan'] = scan

        tensor4d = self.transforms(phase_dict)

        return tensor4d
    
    
    def get_whole_slice(self, idx: int = 0):
        sample = self.dataframe.iloc[idx]
        clean = sample['clean']
        noisy = sample['noise']
        scan = sample['scan']

        clean_phase_keys = [f'phase_0{i}' for i in range(len(clean))]
        noisy_phase_keys = [f'noisy_phase_0{i}' for i in range(len(noisy))]
        phase_dict = {}
        for key_clean, volume_clean, key_noisy, volume_noisy in zip(clean_phase_keys, clean, noisy_phase_keys, noisy):
            phase_dict[key_clean] = volume_clean
            phase_dict[key_noisy] = volume_noisy
        phase_dict['scan'] = scan

        tensor4d = self.scan_transforms(phase_dict)
        return tensor4d['clean'].as_tensor(), tensor4d['noise'].as_tensor()



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
                noisy = sorted(glob(f'{scan_dir}/noise/noisy_phase_**.npy'))
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
            CropBatchTrain(keys=['clean', 'noise'], num_crops=4)
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
            #EnsureBatchDimension(keys=['clean', 'noise']),
            EnsureTyped(keys=['clean', 'noise']),
            CropBatchValidation(keys=['clean', 'noise'])
        ])

        return transforms
    
    @staticmethod
    def get_scan_transforms():
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
            #EnsureBatchDimension(keys=['clean', 'noise']),
            EnsureTyped(keys=['clean', 'noise']),
            MidSlice(keys=['clean', 'noise']),
        ])

        return transforms
    
    @staticmethod
    def get_cache_transforms(destination: str):
        PIXEL_SIZE_IN_MM = 1.16
        HU_MIN, HU_MAX = -1024.0, 3000.0
        clean_keys = [f'phase_0{i}' for i in range(10)]
        noise_keys = [f'noisy_phase_0{i}' for i in range(10)]
        transforms = Compose([
            LoadImaged(keys=clean_keys),
            EnsureChannelFirstd(keys=clean_keys),
            Orientationd(keys=clean_keys, axcodes='RAS'),
            Spacingd(
                keys=clean_keys,
                pixdim=[PIXEL_SIZE_IN_MM, PIXEL_SIZE_IN_MM, 2.5],
                mode=['bilinear'] * len(clean_keys),
            ),
            CustomForegroundCrop(keys=clean_keys, threshold_hu=-800.0),
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
                spatial_size=[32, 32, 32],
                mode='constant',
                constant_values=0.0,
            ),
            # (C, H, W, D) -> (C, D, H, W)
            OrderChannels(keys=clean_keys),
            PadToSquare(keys=clean_keys, pad_value=0.0, multiple_of=16),
            GenerateNoiseScans(keys=clean_keys, dosage=0.25, pixel_size_mm=PIXEL_SIZE_IN_MM),
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
        dataloader = torch.utils.data.DataLoader(dataset=dataset, shuffle=False, num_workers=2, collate_fn=CT4dDataset.cache_collate)
        tqdm_loader = tqdm(dataloader)
        for index, out in enumerate(tqdm_loader):
            keys = list(out.keys())
            shapes = [out[key].shape for key in keys]
            tqdm_loader.set_description(f'Cached batch with index {index}, scans: {keys}, shapes: {shapes}')


