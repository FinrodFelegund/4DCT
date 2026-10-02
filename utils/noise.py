import numpy as np
import torch

from torch_radon import ParallelBeam, FanBeam
from typing import Literal, Tuple
from abc import ABC, abstractmethod

class BaseClassNoise(ABC):

    @abstractmethod
    def add_noise(self, x: torch.Tensor):
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor):
        pass

class SinogramNoise(BaseClassNoise):
    def __init__(
            self,
            image_size: int = 512,
            dosage: float = 0.25,
            pixel_size_mm: float = 1.16,
            i0_full: int = 1e5,
            electronic_std: float = 10,
            n_angles: int = 720,
            hu_min: float = -1024.0,
            hu_max: int = 3000.0,
            mu_water: float = 0.02,
            device: str | torch.device = 'cuda',
            b_type: Literal['Parallel-Beam', 'Fan-Beam'] = 'Parallel-Beam',
            
        ):
        self.image_size = image_size
        self.dosage = dosage
        self.pixel_size_mm = pixel_size_mm
        self.i0_full = i0_full
        self.electronic_std = electronic_std
        self.n_angles = n_angles
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.mu_water = mu_water
        self.device = device
        self.b_type = b_type

        self.src_dist_mm = None
        self.det_dist_mm = None

        self.radon = None
        if self.b_type == 'Parallel-Beam':
            self.radon = ParallelBeam(int(self.image_size * 1.5),
                                    np.linspace(0, np.pi, self.n_angles, endpoint=False))
        elif self.b_type == 'Fan-Beam':
            raise NotImplementedError('Fan-Beam currently not supported')
            s_dist = self.src_dist_mm / self.pixel_size_mm
            d_dist = self.det_dist_mm / self.pixel_size_mm
            magnification = (s_dist + d_dist) / s_dist
            det_count = int(np.ceil(self.image_size * np.sqrt(2) * magnification)) + 2
            self.radon = FanBeam(det_count, np.linspace(0, 2 * np.pi, self.n_angles, endpoint=False),
                                 src_dist=s_dist, det_dist=d_tist)
        else:
            raise ValueError(f'Unsupported beam type, got {self.b_type}, expected one of [Parallel-Beam, Fan-Beam]')


    def norm_to_mu(self, x):
        hu = x * (self.hu_max - self.hu_min) + self.hu_min
        return (self.mu_water * (1.0 + (hu / 1000.0))).clamp_min(0.0)

    def mu_to_norm(self, mu):
        hu = (mu / self.mu_water - 1.0) * 1000
        return (hu - self.hu_min) / (self.hu_max - self.hu_min)

    def forward(self, mu):
        return self.radon.forward(mu) * self.pixel_size_mm

    def backward(self, sino):
        filtered_sino = self.radon.filter_sinogram(sino)
        return self.radon.backward(filtered_sino) / self.pixel_size_mm

    def add_noise(self, sino):
        counts = torch.poisson(self.i0_full * self.dosage * torch.exp(-sino))
        counts = counts + torch.randn_like(counts) * self.electronic_std
        return -torch.log(counts.clamp_min(1.0) / (self.i0_full * self.dosage))

    def __call__(self, x):
        if x.shape[-1] != x.shape[-2] or x.shape[-1] != self.image_size:
            raise ValueError(f'expected square {self.image_size} slices, got {tuple(x.shape)}')

        sino = self.forward(self.norm_to_mu(x))
        clean = self.mu_to_norm(self.backward(sino))
        noisy = self.mu_to_norm(self.backward(self.add_noise(sino)))
        return clean, noisy