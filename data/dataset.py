from torch.utils.data import Dataset
import pandas as pd
from typing import List, Dict
import torch
import os
from glob import glob
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



    
class OrderChannels(MapTransform):
    def __init__(self, keys, allow_missing_keys = False):
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data: Dict):
        keys = list(data.keys())
        for key in keys:
            img = data[key]
            img = img.permute(0, 3, 1, 2)
            data[key] = img

        return data
    
class EnsureChannelDimension(MapTransform):
    def __init__(self, keys, allow_missing_keys = False):
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
    
    def __call__(self, data: Dict):
        keys = list(data.keys())
        for key in keys:
            img = data[key]
            if len(img.shape) != 5:
                img = img.unsqueeze(0)
                data[key] = img

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
        clean = sample['clean']
        noisy = sample['noisy']

        clean_phase_keys = [f'phase_0{i}' for i in range(len(clean))]
        noisy_phase_keys = [f'noisy_phase_0{i}' for i in range(len(noisy))]
        phase_dict = {}
        for key_clean, volume_clean, key_noisy, volume_noisy in zip(clean_phase_keys, clean, noisy_phase_keys, noisy):
            phase_dict[key_clean] = volume_clean
            phase_dict[key_noisy] = volume_noisy

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
            scans = sorted(os.listdir(dataset))
            for scan in scans:
                scan_dir = os.path.join(dataset, scan)
                clean = sorted(glob(f'{scan_dir}/images/phase_**.nii'))
                noisy = sorted(glob(f'{scan_dir}/noise/noise_phase**.nii'))
                data_dict[scan] = {
                    'clean': clean,
                    'noisy': noisy,
                }
        
        return pd.DataFrame(data_dict).transpose()
    
    @staticmethod
    def collate_fn(data: List[dict]):
        gt_data = [tensor[0]['clean'].as_tensor() for tensor in data]
        gt_data = torch.stack(gt_data, dim=0)
        noise_data = [tensor[0]['noise'].as_tensor() for tensor in data]
        noise_data = torch.stack(noise_data, dim=0)
        return gt_data, noise_data
    
    @staticmethod
    def get_train_transforms():
        clean_keys = [f'phase_0{i}' for i in range(10)]
        noise_keys = [f'noisy_phase_0{i}' for i in range(10)]
        transforms = Compose([
            LoadImaged(keys=clean_keys + noise_keys),
            EnsureChannelFirstd(keys=clean_keys + noise_keys),
            Spacingd(
                keys=clean_keys + noise_keys,
                pixdim=[1.16, 1.16, 2.5],
                mode=['bilinear'] * len(clean_keys) * 2,
            ),
            SpatialPadd(
                keys=clean_keys + noise_keys,
                spatial_size=[32, 32, 32],
                mode='constant',
                constant_values=0,
            ),
            OrderChannels(keys=clean_keys),
            ScaleIntensityRanged(
                keys=clean_keys + noise_keys,
                a_min=-1000,
                a_max=1000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
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
            EnsureChannelDimension(keys=['clean', 'noise']),
            EnsureTyped(keys=['clean', 'noise']),

        ])

        return transforms
    
    @staticmethod
    def get_validation_transforms():
        clean_keys = [f'phase_0{i}' for i in range(10)]
        noise_keys = [f'noisy_phase_0{i}' for i in range(10)]
        transforms = Compose([
            LoadImaged(keys=clean_keys + noise_keys),
            EnsureChannelFirstd(keys=clean_keys + noise_keys),
            Spacingd(
                keys=clean_keys + noise_keys,
                pixdim=[1.16, 1.16, 2.5],
                mode=['bilinear'] * len(clean_keys) * 2,
            ),
            SpatialPadd(
                keys=clean_keys + noise_keys,
                spatial_size=[32, 32, 32],
                mode='constant',
                constant_values=0,
            ),
            OrderChannels(keys=clean_keys + noise_keys),
            ScaleIntensityRanged(
                keys=clean_keys + noise_keys,
                a_min=-1000,
                a_max=1000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
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
            EnsureChannelDimension(keys=['clean', 'noise']),
            EnsureTyped(keys=['clean', 'noise']),

        ])

        return transforms