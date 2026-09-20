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
            max_proj: float,
            b_type: Literal['Parallel-Beam', 'Fan-Beam'] = 'Fan-Beam',
            image_size: int = 512,
            dosage: float = 0.25,
            pixel_size_mm: float = 1.16,
            src_dist_mm: float = 595.0,
            det_dist_mm: float = 490.6,
            i0_full = 1e5,
            device: str | torch.device = 'cuda:0',
            
        ):
        
        self.max_proj = max_proj
        self.b_type = b_type
        self.image_size = image_size
        self.det_count = int(image_size * 1.5)
        self.I0 = i0_full * dosage
        self.dosage = dosage
        self.pixel_size_mm = pixel_size_mm

        s_dist = src_dist_mm / pixel_size_mm
        d_dist = det_dist_mm / pixel_size_mm
        

        if self.b_type == 'Parallel-Beam':
            angles = np.linspace(0, np.pi, num=720, endpoint=False)
            self.radon = ParallelBeam(self.det_count, angles)
        
        else:
            angles = np.linspace(0, 2*np.pi, num=720, endpoint=False)
            self.radon = FanBeam(self.det_count, angles)


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

        noisy_sinogram = (poisson * self.dosage) * torch.exp(-sinogram / self.max_proj)
        noisy_sinogram = torch.poisson(noisy_sinogram)
        noisy_sinogram = noisy_sinogram + torch.randn_like(noisy_sinogram) * gaussian
        noisy_sinogram = noisy_sinogram.clamp_min(1.0)
        noisy_sinogram = -torch.log(noisy_sinogram / (poisson * self.dosage)) * self.max_proj
        
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
        sino_noise = self.add_noise(sino, poisson=1e5, gaussian=10.0)
        recon = self.backward(sino)
        recon_noisy = self.backward(sino_noise)
        return recon.clamp(0.0, 1.0), recon_noisy.clamp(0.0, 1.0)


    def compute_max_proj(self, x: torch.Tensor):
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
        
        max_proj = torch.max(sino).reshape((1, -1))
        return max_proj