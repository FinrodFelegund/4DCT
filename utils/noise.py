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
            b_type: Literal['Parallel-Beam', 'Fan-Beam'] = 'Fan-Beam',
            image_size: int = 512,
            dosage: float = 0.5,
            pixel_size_mm: float = 1.16,
            mu_water_per_mm: float = 0.019,
            hu_range: Tuple[float, float] = (-1024.0, 3000.0),
            i0_full = 2e6,
            device: str | torch.device = 'cuda:0'
        ):
        
        self.b_type = b_type
        self.image_size = image_size
        self.det_count = int(image_size * 1.5)
        self.I0 = i0_full * dosage
        self.dosage = dosage
        self.pixel_size_mm = pixel_size_mm
        self.mu_water_per_mm = mu_water_per_mm
        self.hu_min, self.hu_max = hu_range
        self.device = torch.device(device)

        if self.b_type == 'Parallel-Beam':
            angles = np.linspace(0, np.pi, num=720, endpoint=False)
            self.radon = ParallelBeam(self.det_count, angles)
        
        else:
            angles = np.linspace(0, 2*np.pi, num=720, endpoint=False)
            self.radon = FanBeam(self.det_count, angles)

        self.device = torch.device('cuda:0')

    def _norm_to_mu(self, x: torch.Tensor):
        hu = x * (self.hu_max - self.hu_min) + self.hu_min
        mu_per_mm = self.mu_water_per_mm * (hu / 1000.0 + 1.0)
        mu_per_mm = mu_per_mm.clamp(min=0.0)
        return mu_per_mm * self.pixel_size_mm
    
    def _mu_to_norm(self, mu_px: torch.Tensor) -> torch.Tensor:
        """mu per pixel -> [0,1] normalized image (inverse of _norm_to_mu)."""
        mu_per_mm = mu_px / self.pixel_size_mm
        hu = (mu_per_mm / self.mu_water_per_mm - 1.0) * 1000.0
        x = (hu - self.hu_min) / (self.hu_max - self.hu_min)
        return x.clamp(0.0, 1.0)



    def forward(self, x: torch.Tensor):
        """
        Radon forward expects data to have shape [d1, ..., dn, self.image_size, self.image_size].
        Assumes Images to be in shape [D, H, W]
        """
        
        sinogram = self.radon.forward(x)

        return sinogram
    
    def backward(self, x: torch.Tensor):
        """Return filtered backprojection of the Radon Transform."""
        x = self.radon.filter_sinogram(x)
        x = self.radon.backward(x)

        return x

    def add_noise(self, sinogram: torch.Tensor, poisson = None, gaussian = None):
        """Returns Poisson noise altered sinograms. https://github.com/CERN/TIGRE/blob/master/Python/tigre/utilities/CTnoise.py"""

        if poisson is not None:
            if not np.isscalar(poisson):
                raise ValueError(f'Poisson should be a scalar, got {type(poisson)} instead')
        else:
            poisson = torch.ceil(torch.log2(torch.max(torch.abs(sinogram))))

        if gaussian is not None:
            if not isinstance(gaussian, float):
                raise ValueError(f'Gaussian should be type float, got {type(gaussian)} instead')
        else: 
            gaussian = 10.0
        """
        max_proj = np.max(projections)
        projections = Poisson * np.exp(-projections / max_proj)

        projections = RNG.add_noise(projections, Gaussian[0], Gaussian[1])

        projections = -np.log(projections / Poisson) * max_proj
        projections = np.float32(projections)
        """

        max_proj = torch.amax(sinogram, dim=(1,2), keepdim=True)
        noisy_sinogram = (poisson * self.dosage) * torch.exp(-sinogram / max_proj)
        noisy_sinogram = torch.poisson(noisy_sinogram)
        noisy_sinogram = noisy_sinogram + torch.randn_like(noisy_sinogram) * gaussian
        noisy_sinogram = -torch.log(noisy_sinogram / (poisson * self.dosage)) * max_proj
        
        return noisy_sinogram

    
    def __call__(self, x: torch.Tensor):
        if x.shape[-1] != x.shape[-2]:
            raise ValueError(
                f'SinogramNoise expects square slices, got {tuple(x.shape)}. '
                'Pad H/W to a common size before applying noise.'
            )
        if x.shape[-1] != self.image_size:
            raise ValueError(
                f'SinogramNoise was built for image_size={self.image_size}, '
                f'got {x.shape[-1]}. Construct a matching instance.'
            )
        
        #mu = self._norm_to_mu(x)
        sino = self.forward(x)
        sino = self.add_noise(sino, poisson=1e5, gaussian=10.0)
        recon = self.backward(sino)
        return recon.clamp(0.0, 1.0)


    
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