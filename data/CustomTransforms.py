from monai.transforms import MapTransform
from typing import Dict
import numpy as np
from pathlib import Path
import torch
from utils.noise import SinogramNoise
import zlib
import json
    
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

    
class CacheBatch(MapTransform):
    def __init__(self, keys, allow_missing_keys = False):
        
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data: Dict):
        dest_path = Path(data['destination'])
        clean = dest_path / 'images'
        clean.mkdir(parents=True, exist_ok=True)
        noise = dest_path /  'noise'
        noise.mkdir(parents=True, exist_ok=True)

    
        for key in self.keys:
            if key not in data:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f'Key {key} missing from data')
            target_dir = noise if key.startswith('noise_') else clean
            file_path = target_dir / f'{key}.npy'

            img = torch.as_tensor(data[key]).detach().cpu().numpy().astype(np.float32)
            data[key] = img.shape
            np.save(file_path, img)

        (dest_path / 'simulation.json').write_text(json.dumps(
            {'scan': str(data['scan']), 'i0_full': data.get('i0_full')},
            indent=2
        ))
    
        return data



class CropBatchTrain(MapTransform):
    def __init__(self, keys,
                 allow_missing_keys = False,
                 num_crops=8,
                 patch_size=(32, 64, 64),
                 margin=8,
                 body_thresh=0.018,
                 min_body_fraction=0.5,
                 attempts=20):
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.num_crops = num_crops
        self.patch_size = patch_size
        self.margin = margin
        self.body_thresh = body_thresh
        self.min_body_fraction = min_body_fraction
        self.attempts = attempts

    def __call__(self, data: Dict):
        pd, ph, pw = self.patch_size
        size_d, md = pd + 2 * self.margin, self.margin
        size_h, mh = ph + 2 * self.margin, self.margin
        size_w, mw = pw + 2 * self.margin, self.margin
        _, _, D, H, W = data['clean'].shape
        if D < size_d or H < size_h or W < size_w:
            raise ValueError(f'volume {D}x{H}x{W} smaller than crop {self.patch_size}')
        
        stack_clean, stack_noise = [], []
        for _ in range(self.num_crops):
            for _ in range(self.attempts):
                d = np.random.randint(0, D - size_d + 1)
                y = np.random.randint(0, H - size_h + 1)
                x = np.random.randint(0, W - size_w + 1)
                crop = data['clean'][:, :, d:d + size_d, y:y + size_h, x:x + size_w]
                core = crop[..., md:size_d - md, mh:size_h - mh, mw:size_w - mw]
                if (core > self.body_thresh).float().mean() >= self.min_body_fraction:
                    break
            
            stack_clean.append(crop)
            stack_noise.append(data['noise'][:, :, d:d + size_d, y:y + size_h, x:x + size_w])

        data['clean'] = torch.stack(stack_clean, dim=0)
        data['noise'] = torch.stack(stack_noise, dim=0)

        return data

class CropBatchValidation(MapTransform):
    def __init__(self, keys,
                 allow_missing_keys=False,
                 source_key='clean',
                 num_crops_per_dim=4,
                 patch_size=(32, 64, 64),
                 margin=8,
                 body_thresh=0.018,
                 min_body_fraction=0.5):
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.source_key = source_key
        self.num_crops_per_dim = num_crops_per_dim
        self.patch_size = patch_size
        self.margin = margin
        self.body_thresh = body_thresh
        self.min_body_fraction = min_body_fraction

    def _core_stats(self, dim_size, ps):
        usable = dim_size - 2 * self.margin
        n = max(1, min(self.num_crops_per_dim, usable // ps))
        first = self.margin + (usable - n * ps) // 2
        return [first + i * ps for i in range(n)]

    def __call__(self, data: Dict):
        (pd, ph, pw), m = self.patch_size, self.margin
        ref = data[self.source_key]
        _, _, D, H, W = ref.shape
        if D < pd + 2 * m or H < ph + 2 * m or W < pw + 2 * m:
            raise ValueError(f'volume {D}x{H}x{W} smaller than crop {pd}x{ph}x{pw}')

        positions = []
        for d in self._core_stats(D, pd):
            for h in self._core_stats(H, ph):
                for w in self._core_stats(W, pw):
                    core = ref[:, :, d:d + pd, h:h + ph, w:w + pw]
                    if (core > self.body_thresh).float().mean() >= self.min_body_fraction:
                        positions.append((d - m, h - m, w - m))

        if not positions:
            raise ValueError('No validation crop contains enough body')

        for key in self.keys:
            if key not in data:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f'Key {key} missing from data')
            img = data[key]
            data[key] = torch.stack(
                [img[:, :, d:d + pd + 2 * m, h:h + ph + 2 * m, w:w + pw + 2 * m]
                 for d, h, w in positions], dim=0)
        return data
    

class GenerateNoiseBatch(MapTransform):
    def __init__(self, keys, allow_missing_keys=False, dosage: float = 0.25,
                 i0_full: float = 1e5, i0_by_scan: dict | None = None,
                 pixel_size_mm: float = 1.16,
                 device: str | torch.device = 'cuda:0',
                 slice_size: int = 16):
        
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.dosage = dosage
        self.i0_full = i0_full
        self.i0_by_scan = i0_by_scan
        self.pixel_size_mm = pixel_size_mm
        self.device = torch.device(device)
        self.slice_size = slice_size
        self.noise_model = None
        self.noise_model_size = None

    def get_noise_model(self, image_size: int, i0: float):
        if self.noise_model_size != image_size:
            self.noise_model = SinogramNoise(
                image_size=image_size, dosage=self.dosage, i0_full=i0,
                pixel_size_mm=self.pixel_size_mm, b_type='Parallel-Beam',
                electronic_std=10.0, n_angles=720,
                hu_min=-1024.0, hu_max=3000.0
            )

            self.noise_model_size = image_size

        self.noise_model.i0_full = i0
        return self.noise_model
        

    def __call__(self, data: Dict):
        scan = str(data['scan'])
        if self.i0_by_scan is not None:
            if scan not in self.i0_by_scan:
                raise KeyError(f'{scan}: no fitted photon count')
            i0 = float(self.i0_by_scan[scan])
        else:
            i0 = float(self.i0_full)

        data['i0_full'] = i0
        torch.manual_seed(zlib.crc32(scan.encode()))

        for key in self.keys:
            if key not in data:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f'Key {key} is missing from data')

            img = torch.as_tensor(data[key])
            C, D, H, W = img.shape
            noise_model = self.get_noise_model(H, i0)
            

            step = C * D if self.slice_size is None else self.slice_size
            slices = img.reshape(C * D, H, W)
            recon, noisy = [], []
            with torch.no_grad():
                for s in range(0, C * D, step):
                    r, n = noise_model(slices[s:s + step].to(self.device))
                    recon.append(r.cpu())
                    noisy.append(n.cpu())

            data[key] = torch.cat(recon).reshape(C, D, H, W)
            data[f'noise_{key}'] = torch.cat(noisy).reshape(C, D, H, W)
                    
        return data
    
class ForegroundBBox(MapTransform):

    def __init__(self, keys, source_key='clean', thresh=0.056, margin=8, min_size=32):
        self.keys = keys
        self.source_key = source_key
        self.thresh = thresh
        self.margin = margin
        self.min_size = min_size
        
    def __call__(self, data):
        src = torch.as_tensor(data[self.source_key])       # (1,10,D,H,W)
        vol = src.reshape(-1, *src.shape[-3:])             # (N,D,H,W)
        fg  = (vol > self.thresh).any(dim=0)               # (D,H,W)
        hw  = fg.any(dim=0)                                # (H,W)  keep full D
        ys, xs = torch.where(hw.any(1))[0], torch.where(hw.any(0))[0]
        if len(ys) == 0 or len(xs) == 0:
            return data
        H, W = fg.shape[-2:]
        y0, y1 = max(int(ys[0])-self.margin, 0), min(int(ys[-1])+1+self.margin, H)
        x0, x1 = max(int(xs[0])-self.margin, 0), min(int(xs[-1])+1+self.margin, W)
        # never let the bbox fall below patch size
        def pad(a, b, n, hi):
            if b - a >= n: return a, b
            c = (a + b) // 2; a = min(max(c - n//2, 0), hi - n); return a, a + n
        y0, y1 = pad(y0, y1, self.min_size, H)
        x0, x1 = pad(x0, x1, self.min_size, W)
        for k in self.keys:
            if k in data:
                data[k] = data[k][..., y0:y1, x0:x1]
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
    def __init__(self, keys, allow_missing_keys=False, context=8):
        """Visualization during training"""
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.context = context

    def __call__(self, data: Dict):
        ref_key = next(k for k in self.keys if k in data)
        D = data[ref_key].shape[2]
        mid = D // 2
        lo, hi = max(mid - self.context, 0), min(mid + self.context + 1, D)

        for key in self.keys:
            if key not in data:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f'Key {key} missing from data')
            data[key] = data[key][:, :, lo:hi].unsqueeze(0)
        return data


class AssertPhasesAligned(MapTransform):
    def __init__(self, keys, allow_missing_keys = False, ref_key = 'phase_01'):
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.ref_key = ref_key

    def __call__(self, data: Dict):
        ref_shape = tuple(data[self.ref_key].shape)
        for key in self.keys:
            if key not in data:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f'Key {key} missing from data')

            if tuple(data[key].shape) != ref_shape:
                raise ValueError(f'{data.get("scan")}: {key} has shape {tuple(data[key].shape)},   '
                                 f'expected {ref_shape}')



        return data

class ApplyHUOffset(MapTransform):
    def __init__(self, keys, offsets, allow_missing_keys = False):
        self.keys = keys
        self.offsets = offsets
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data: Dict):
        offset = self.offsets.get(str(data['scan']), 0.0)
        if offset:
            for key in self.keys:
                if not key in data:
                    if self.allow_missing_keys:
                        continue
                    raise KeyError(f'Key {key} missing from train data')
                data[key] = data[key] + offset

        return data



