from monai.transforms import MapTransform, CropForegroundd
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
    def __init__(self, keys, allow_missing_keys = False, destination = None):
        if not destination:
            raise ValueError(f'Expected destination path but got {destination}')
        
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

    
        for key in self.keys:
            if key not in data:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f'Key {key} missing from data')
            target_dir = noisy if key.startswith('noisy_') else clean
            file_path = target_dir / f'{key}.npy'

            img = torch.as_tensor(data[key]).detach().cpu().numpy().astype(np.float16)
            data[key] = img
            np.save(file_path, img)
                
        return data



class CropBatchTrain(MapTransform):
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

class CropBatchValidation(MapTransform):
    def __init__(self, keys, allow_missing_keys = False, num_crops_per_dim=8, patch_size=32):
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.num_crops_per_dim = num_crops_per_dim
        self.patch_size = patch_size

    def _centered_offsets(self, dim_size: int) -> list:
        n = min(self.num_crops_per_dim, dim_size // self.patch_size)
        n = max(1, n)
        extent = n * self.patch_size
        start = max((dim_size - extent) // 2, 0)
        offsets = [
            min(start + i * self.patch_size, dim_size - self.patch_size)
            for i in range(n)
        ]

        return offsets


    def __call__(self, data: Dict):
        ref_key = next(k for k in self.keys if k in data)
        _, _, D, H ,W = data[ref_key].shape

        d_offs = self._centered_offsets(D)
        h_offs = self._centered_offsets(H)
        w_offs = self._centered_offsets(W)
        ps = self.patch_size

        for key in self.keys:
            if key not in data:
                if self.allow_missing_keys:
                    continue
                else:
                    raise KeyError(f'Key {key} missing from data')
            img = data[key]
            patches = [
                img[:, :, d:d+ps, h:h+ps, w:w+ps]
                for d in d_offs
                for h in h_offs
                for w in w_offs
            ]

            data[key] = torch.stack(patches, dim=0)
        
        return data
    

class GenerateNoiseScans(MapTransform):
    def __init__(self, keys, allow_missing_keys = False, dosage: float = 0.25, pixel_size_mm: float = 1.16, device: str | torch.device = 'cuda:0', slice_size: int = 16):
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.dosage = dosage
        self.pixel_size_mm = pixel_size_mm
        self.device = torch.device(device)
        self.noise_model = None
        self._noise_model_size = None
        self.slice_size = slice_size

    def _get_noise_model(self, image_size: int):
        if self._noise_model_size != image_size:
            self.noise_model = None
            torch.cuda.empty_cache()
            self.noise_model = SinogramNoise(
                b_type='Fan-Beam',
                image_size=image_size,
                dosage=self.dosage,
                pixel_size_mm=self.pixel_size_mm,
                device=self.device,
            )

            self._noise_model_size = image_size
        
        return self.noise_model
        

    def __call__(self, data: Dict):
        for key in self.keys:
            if key not in data:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f'Key {key} missing from data')
            
            img = torch.as_tensor(data[key]).to(self.device)
            C, D, H, W = img.shape
            noise_model = self._get_noise_model(H)

            with torch.no_grad():
                slices = img.reshape(C*D, H, W)
                noisy = torch.empty_like(slices)
                for start in range(0, C * D, self.slice_size):
                    end = min(start + self.slice_size, C * D)
                    slab = slices[start:end].to(self.device)
                    noisy[start:end] = noise_model(slab).cpu()
                    del slab
            
            data[f'noisy_{key}'] = noisy.reshape(C, D, H, W)
            del slices, noisy

        torch.cuda.empty_cache()

        return data
    
class CustomForegroundCrop(MapTransform):
    def __init__(self, keys, allow_missing_keys = False, threshold_hu: float = -800.0):
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.threshold_hu = threshold_hu

    def __call__(self, data: Dict):
        present = [k for k in self.keys if k in data]
        if not present:
            if self.allow_missing_keys:
                return data
            raise KeyError(f'None of {self.keys} found in data')


        union = torch.stack([torch.as_tensor(data[k]) for k in present], dim=0)
        union = union.max(dim=0).values
        data['_fg_source'] = union
     

        cropper =  CropForegroundd(
            keys=present,
            source_key='_fg_source',
            select_fn=lambda x: x > self.threshold_hu,
            start_coord_key=None,
            end_coord_key=None,
        )
        data = cropper(data)
        del data['_fg_source']

        return data
    
class PadToSquare(MapTransform):
    def __init__(self, keys, allow_missing_keys = False, pad_value: float = 0.0, multiple_of: int = 16):
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.pad_value = pad_value
        self.multiple_of = multiple_of

    def __call__(self, data: Dict):
        present = [k for k in self.keys if k in data]
        if not present:
            return data
        
        target = 0
        for k in present:
            H, W = data[k].shape[-2:]
            target = max(target, H, W)
        if self.multiple_of:
            target = int(np.ceil(target / self.multiple_of) * self.multiple_of)

        for k in present:
            img = data[k]
            H, W = img.shape[-2:]
            dh, dw = target - H, target - W
            pad = [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2]
            data[k] = torch.nn.functional.pad(img, pad, mode='constant', value=self.pad_value)

        return data
    

class MidSlice(MapTransform):
    def __init__(self, keys, allow_missing_keys = False):
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data: Dict):
        ref_key = next(k for k in self.keys if k in data)
        _, T, D, H ,W = data[ref_key].shape

        for key in self.keys:
            if key not in data:
                if self.allow_missing_keys:
                    continue
                else:
                    raise KeyError(f'Key {key} missing from data')
            img = data[key]
            mid_t = T // 2
            mid_d = D // 2
            data[key] = img[:, mid_t:mid_t+1, mid_d:mid_d+1, :, :].unsqueeze(0)

        return data

