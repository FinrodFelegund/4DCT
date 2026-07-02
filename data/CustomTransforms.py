from monai.transforms import MapTransform
from typing import Dict
import numpy as np
from pathlib import Path
import torch
from utils.noise import SinogramNoise


    
class OrderChannels(MapTransform):
    def __init__(self, keys, allow_missing_keys = False):
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data: Dict):
        keys = list(data.keys())
        for key in keys:
            if key in self.keys:
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
            if key in self.keys:
                img = data[key]
                if len(img.shape) != 5:
                    img = img.unsqueeze(0)
                    data[key] = img

        return data
    
class EnsureBatchDimension(MapTransform):
    def __init__(self, keys, allow_missing_keys = False):
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data: Dict):
        keys = list(data.keys())
        for key in keys:
            if key in self.keys:
                img = data[key]
                if len(img) != 6:
                    img = img.unsqueeze(0)
                    data[key] = img

        return data
    
class CacheDataSet(MapTransform):
    def __init__(self, keys, destination, allow_missing_keys = False):
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.destination = destination

    def __call__(self, data: Dict):
        path = Path(self.destination)
        scan = Path(data['scan'])
        clean = Path(path / scan / 'images')
        clean.mkdir(parents=True, exist_ok=True)
        noisy = Path(path / scan / 'noise')
        noisy.mkdir(parents=True, exist_ok=True)

        keys = list(data.keys())
        for key in keys:
            if key in self.keys:
                if 'noisy' in key:
                    file_path = Path(noisy / f'{key}.npy')
                else:
                    file_path = Path(clean / f'{key}.npy')

                img = data[key].numpy().astype(np.float16)
                data[key] = img
                np.save(file_path, img)
                
        return data



class CropBatch(MapTransform):
    def __init__(self, keys, allow_missing_keys = False, num_crops=4, patch_size = 32, min_std = 0.02):
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.num_crops = num_crops
        self.patch_size = patch_size
        self.min_std = min_std

    def __call__(self, data: Dict):
        _, _, D, H, W = data['clean'].shape
        img_clean = data['clean']
        img_noise = data['noise']
        stack_clean, stack_noise = [], []
        for _ in range(self.num_crops):
            for attempt in range(10):
                x = np.random.randint(0, W - self.patch_size + 1)
                y = np.random.randint(0, H - self.patch_size + 1)
                d = np.random.randint(0, D - self.patch_size + 1)
                patch = img_clean[:, :, d:d+self.patch_size, y:y+self.patch_size, x:x+self.patch_size]
                if patch.std() > self.min_std:
                    break
            
            stack_clean.append(patch)
            stack_noise.append(img_noise[:, :, d:d+self.patch_size, y:y+self.patch_size, x:x+self.patch_size])

        data['clean'] = torch.stack(stack_clean, dim=0)
        data['noise'] = torch.stack(stack_noise, dim=0)

        return data
    

class GenerateNoiseScans(MapTransform):
    def __init__(self, keys, allow_missing_keys = False):
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data: Dict):
        keys = list(data.keys())

        for key in keys:
            if key in self.keys:
                img = data[key]
                C, D, H, W = img.shape
                img = img.reshape(C * D, H, W)
                s_noise = SinogramNoise(b_type='Fan-Beam', image_size=H, dosage=0.25)
                noise_img = s_noise(data[key]).reshape(C, D, H, W)
                data[f'noise_{key}'] = noise_img

        return data

