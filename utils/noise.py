import numpy as np
import torch

from torch_radon import ParallelBeam, FanBeam
from typing import Literal
from abc import ABC, abstractmethod

class BaseClassNoise(ABC):

    @abstractmethod
    def add_noise(self, x: torch.Tensor):
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor):
        pass

class SinogramNoise(BaseClassNoise):
    def __init__(self, b_type=Literal['Parallel-Beam', 'Fan-Beam'], image_size: int = 512, dosage: float = 0.5):
        self.b_type = b_type
        self.det_count = int (image_size * 1.5)
        self.I0 = 1e5 * dosage
        self.radon = None

        if self.b_type == 'Parallel-Beam':
            angles = np.linspace(0, np.pi, num=720, endpoint=False)
            self.radon = ParallelBeam(self.det_count, angles)
        
        else:
            angles = np.linspace(0, 2*np.pi, num=720, endpoint=False)
            self.radon = FanBeam(self.det_count, angles)

        self.device = torch.device('cuda:1')

    def forward(self, x: torch.Tensor):
        """
        Radon forward expects data to have shape [d1, ..., dn, self.image_size, self.image_size].
        Assumes Images to be in shape [H, W, Z]
        """
        
        x = x.permute(2, 0, 1,)
        sinogram = self.radon.forward(x)

        return sinogram
    
    def backward(self, x: torch.Tensor):
        """Return filtered backprojection of the Radon Transform."""
        x = self.radon.filter_sinogram(x)
        x = self.radon.backward(x)
        x = x.permute(1, 2, 0)
        return x

    def add_noise(self, x: torch.Tensor, max_attenuation: float = 5.0):
        """Returns Poisson noise added sinograms."""

        current_max = x.max()
        x = (x / current_max) * max_attenuation
        x = self.I0 * torch.exp(-x)
        x = torch.poisson(x)
        x = torch.clip(x, min=1.0)
        x = -torch.log(x / self.I0)
        x = (x / (max_attenuation + 1e-8)) * current_max

        return x
    
    def __call__(self, x: torch.Tensor):
        x = x.squeeze(0)
        x = x.to(device=self.device)
        x = self.forward(x)
        x = self.add_noise(x)
        x = self.backward(x)
        x = x.clamp(min=0.0, max=1.0).cpu()
        x = x.unsqueeze(0)
        return x

    
class ImageNoise(BaseClassNoise):
    def __init__(self, noise_level: float = 0.1):
        self.noise_level = noise_level

    def add_noise(self, x: torch.Tensor):
        intensity_variance = (1.0 - x) * self.noise_level
        noise = torch.randn_like(x) * intensity_variance
        background_noise = torch.randn_like(x) * (self.noise_level * 0.5)
        x = x + noise + background_noise
        return x
    
    def __call__(self, x: torch.Tensor):
        x = self.add_noise(x)
        return x.clamp(min=0.0, max=1.0)