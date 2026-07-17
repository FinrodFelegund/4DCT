
"""
Trainable spatio-temporal bilateral filtering (pure PyTorch).
 
Implements the filters from the SPIE abstract:
  - SpatialFilter:  3D bilateral filter (Eq. 1-4), sigmas sx, sy, sz, r
  - TemporalFilter: 1D bilateral filter along T (Eq. 5-7), sigmas t, r
  - FilterBank:     spatial stages followed by temporal stages, in series
 
Style notes:
  - Sigmas are plain nn.Parameters, like the reference CUDA layer. Nothing
    stops them from collapsing toward zero during training; watch the logs.
  - The window covers ~5 sigma per axis (half window = ceil(2.5 * sigma)),
    matching the convention stated in Wagner et al.
  - Gradients come from autograd through unfold. Readable, but memory heavy:
    each spatial stage materialises a [D, H, W, kz*ky*kx] patch tensor.
  - Since the center voxel always contributes weight 1, the kernel sum is
    >= 1 and the normalisation needs no epsilon.
"""
 
import math
 
import torch
 
 
def half_window(sigma, minimum=1, multiplier=2.5):
    """Half window size so the full window covers ~5 sigma (>98% mass)."""
    return max(minimum, int(math.ceil(multiplier * abs(float(sigma)))))
 
 
class SpatialFilter(torch.nn.Module):
    """3D trainable bilateral filter. Input: [B, C, D, H, W].
 
    sigma_sx acts along W, sigma_sy along H, sigma_sz along D.
    """
 
    def __init__(self, sigma_sx, sigma_sy, sigma_sz, sigma_r, border_mode='replicate'):
        super(SpatialFilter, self).__init__()
 
        self.sigma_sx = torch.nn.Parameter(torch.tensor(float(sigma_sx)))
        self.sigma_sy = torch.nn.Parameter(torch.tensor(float(sigma_sy)))
        self.sigma_sz = torch.nn.Parameter(torch.tensor(float(sigma_sz)))
        self.sigma_r = torch.nn.Parameter(torch.tensor(float(sigma_r)))
        self.border_mode = border_mode
        self.kernel_size = None  # (kz, ky, kx), set on forward
 
    def _get_sigmas(self):
        return self.sigma_sx, self.sigma_sy, self.sigma_sz, self.sigma_r
 
    def _update_kernel_size(self):
        sigma_sx, sigma_sy, sigma_sz, _ = self._get_sigmas()
        kz = 2 * half_window(sigma_sz) + 1
        ky = 2 * half_window(sigma_sy) + 1
        kx = 2 * half_window(sigma_sx) + 1
        self.kernel_size = (kz, ky, kx)
 
    def get_kernel_size(self):
        return self.kernel_size
 
    def _compute_spatial_kernel(self, device, dtype):
        sigma_sx, sigma_sy, sigma_sz, _ = self._get_sigmas()
        kz, ky, kx = self.kernel_size
 
        dz = torch.arange(-(kz // 2), kz // 2 + 1, device=device, dtype=dtype)
        dy = torch.arange(-(ky // 2), ky // 2 + 1, device=device, dtype=dtype)
        dx = torch.arange(-(kx // 2), kx // 2 + 1, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(dz, dy, dx, indexing='ij')
 
        exponent = (
            -(xx ** 2) / (2 * sigma_sx ** 2) +
            -(yy ** 2) / (2 * sigma_sy ** 2) +
            -(zz ** 2) / (2 * sigma_sz ** 2)
        )
        return torch.exp(exponent).reshape(-1)
 
    def _compute_range_kernel(self, center_values, neighbor_values):
        _, _, _, sigma_r = self._get_sigmas()
        difference = (center_values - neighbor_values) ** 2
        return torch.exp(-difference / (2 * sigma_r ** 2))
 
    def forward(self, x: torch.Tensor):
        if not x.dim() == 5:
            raise ValueError(f'Input must be a 5D tensor. Got shape: {x.shape}')
 
        self._update_kernel_size()
        kz, ky, kx = self.kernel_size
        pad = [kx // 2] * 2 + [ky // 2] * 2 + [kz // 2] * 2
 
        B, C, D, H, W = x.shape
        output = torch.empty_like(x)
        spatial_kernel = self._compute_spatial_kernel(x.device, x.dtype)
        spatial_kernel = spatial_kernel.view(1, 1, 1, 1, 1, -1)
 
        for i in range(B):
            x_s = x[i:i + 1]
            x_pad = torch.nn.functional.pad(x_s, pad, mode=self.border_mode)
            patches = x_pad.unfold(2, kz, 1).unfold(3, ky, 1).unfold(4, kx, 1)
            patches = patches.contiguous().view(1, C, D, H, W, -1)
 
            range_kernel = self._compute_range_kernel(x_s.unsqueeze(-1), patches)
            kernel = spatial_kernel * range_kernel
            kernel = kernel / kernel.sum(dim=-1, keepdim=True)
            output[i:i + 1] = (patches * kernel).sum(dim=-1)
            del patches, kernel
 
        return output
 
 
class TemporalFilter(torch.nn.Module):
    """1D trainable bilateral filter along T. Input: [B, C, T, D, H, W]."""
 
    def __init__(self, sigma_t, sigma_r, border_mode='replicate'):
        super(TemporalFilter, self).__init__()
 
        self.sigma_t = torch.nn.Parameter(torch.tensor(float(sigma_t)))
        self.sigma_r = torch.nn.Parameter(torch.tensor(float(sigma_r)))
        self.border_mode = border_mode
        self.kernel_size = None
 
    def _get_sigmas(self):
        return self.sigma_t, self.sigma_r
 
    def _update_kernel_size(self):
        sigma_t, _ = self._get_sigmas()
        self.kernel_size = 2 * half_window(sigma_t) + 1
 
    def get_kernel_size(self):
        return self.kernel_size
 
    def _compute_temporal_kernel(self, device, dtype):
        sigma_t, _ = self._get_sigmas()
        coords = torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1,
                              device=device, dtype=dtype)
        return torch.exp(-(coords ** 2) / (2 * sigma_t ** 2))
 
    def _compute_range_kernel(self, center_values, neighbor_values):
        _, sigma_r = self._get_sigmas()
        difference = (center_values - neighbor_values) ** 2
        return torch.exp(-difference / (2 * sigma_r ** 2))
 
    def forward(self, x: torch.Tensor):
        if not x.dim() == 6:
            raise ValueError(f'Input must be a 6D tensor. Got shape {x.shape}')
 
        self._update_kernel_size()
        pad = self.kernel_size // 2
        output = torch.empty_like(x)
        temporal_kernel = self._compute_temporal_kernel(x.device, x.dtype)
        temporal_kernel = temporal_kernel.view(1, 1, 1, -1)
 
        for i in range(x.shape[0]):
            x_s = x[i:i + 1]
            B, C, T, D, H, W = x_s.shape
            # fold every spatial position into the batch: 1D filtering along T
            x_s = x_s.permute(0, 3, 4, 5, 1, 2).contiguous().view(B * D * H * W, C, T)
            x_pad = torch.nn.functional.pad(x_s, [pad] * 2, mode=self.border_mode)
            patches = x_pad.unfold(2, self.kernel_size, 1)
 
            range_kernel = self._compute_range_kernel(x_s.unsqueeze(-1), patches)
            kernel = temporal_kernel * range_kernel
            kernel = kernel / kernel.sum(dim=-1, keepdim=True)
            filtered = (patches * kernel).sum(dim=-1)
            output[i:i + 1] = filtered.view(B, D, H, W, C, T).permute(0, 4, 5, 1, 2, 3)
            del patches, kernel
 
        return output
 
 
class FilterBank(torch.nn.Module):
    """Spatial stages followed by temporal stages, applied in series.
 
    num_spatial=3, num_temporal=1 reproduces the spatio-temporal pipeline
    from the abstract; num_temporal=0 gives the purely spatial baseline.
    """
 
    def __init__(self, num_spatial, num_temporal, border_mode='replicate'):
        super(FilterBank, self).__init__()
        self.spatial_kernels = torch.nn.ModuleList([SpatialFilter(
            sigma_sx=1.0,
            sigma_sy=1.0,
            sigma_sz=1.0,
            sigma_r=0.5,
            border_mode=border_mode,
        ) for _ in range(num_spatial)])
 
        self.temporal_kernels = torch.nn.ModuleList([TemporalFilter(
            sigma_t=1.0,
            sigma_r=0.5,
            border_mode=border_mode,
        ) for _ in range(num_temporal)])
 
    def forward(self, x):
        B, C, T, D, H, W = x.shape
 
        x = x.view(B * T, C, D, H, W)
        for stage in self.spatial_kernels:
            x = stage(x)
 
        x = x.view(B, C, T, D, H, W)
        for stage in self.temporal_kernels:
            x = stage(x)
        return x
 
    def get_sigmas_by_type(self):
        data = {
            'spatial_sx': [], 'spatial_sy': [], 'spatial_sz': [], 'spatial_r': [],
            'temporal_t': [], 'temporal_r': [],
        }
 
        for spatial_kernel in self.spatial_kernels:
            sx, sy, sz, sr = spatial_kernel._get_sigmas()
            data['spatial_sx'].append(sx.item())
            data['spatial_sy'].append(sy.item())
            data['spatial_sz'].append(sz.item())
            data['spatial_r'].append(sr.item())
 
        for temporal_kernel in self.temporal_kernels:
            st, sr = temporal_kernel._get_sigmas()
            data['temporal_t'].append(st.item())
            data['temporal_r'].append(sr.item())
 
        return data
 
    def get_kernel_size(self):
        kernel_sizes = {}
        for i, spatial_kernel in enumerate(self.spatial_kernels):
            kernel_sizes[f'spatial_kernel_{i + 1}_kernel_size'] = spatial_kernel.get_kernel_size()
 
        for i, temporal_kernel in enumerate(self.temporal_kernels):
            kernel_sizes[f'temporal_kernel_{i + 1}_kernel_size'] = temporal_kernel.get_kernel_size()
 
        return kernel_sizes
 
    def __repr__(self):
        num_spatial = len(self.spatial_kernels)
        num_temporal = len(self.temporal_kernels)
 
        return f'Number of spatial kernels: {num_spatial} | Number of temporal kernels: {num_temporal}'