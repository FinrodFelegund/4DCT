import torch
import numpy as np

def inverse_softplus(y):
    y = max(y, 1e-6)
    return np.log(np.exp(y) - 1)

class BoundedSoftplus(torch.nn.Module):
    def __init__(self, beta=1.0, threshold=6.0):
        super(BoundedSoftplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x):
        x = torch.nn.functional.softplus(x, beta=self.beta)
        x = torch.clamp(x, max=self.threshold)
        return x + 1e-6


class SpatialFilter(torch.nn.Module):
    
    def __init__(self, sigma_sx, sigma_sy, sigma_sz, sigma_r, device):
        super(SpatialFilter, self).__init__()



        self.kernel_size = None
        self.pad = None
        self.sigma_sx = torch.nn.Parameter(torch.tensor(inverse_softplus(sigma_sx), dtype=torch.float32)).to(device)
        self.sigma_sy = torch.nn.Parameter(torch.tensor(inverse_softplus(sigma_sy), dtype=torch.float32)).to(device)
        self.sigma_sz = torch.nn.Parameter(torch.tensor(inverse_softplus(sigma_sz), dtype=torch.float32)).to(device)
        self.sigma_r = torch.nn.Parameter(torch.tensor(inverse_softplus(sigma_r), dtype=torch.float32)).to(device)

        self.spatial_activation = BoundedSoftplus(threshold=1.5)
        self.range_activation = BoundedSoftplus(threshold=6.0)
        self.device = device

    def _get_activated_sigmas(self):
        sigma_sx = self.spatial_activation(self.sigma_sx)
        sigma_sy = self.spatial_activation(self.sigma_sy)
        sigma_sz = self.spatial_activation(self.sigma_sz)
        sigma_r = self.range_activation(self.sigma_r)
        return sigma_sx, sigma_sy, sigma_sz, sigma_r 

    def _update_kernel_size(self):
        sigma_sx, sigma_sy, sigma_sz, _ = self._get_activated_sigmas()
        max_sigma = max([sigma_sx.item(), sigma_sy.item(), sigma_sz.item()])
        kernel_size = int(2 * np.ceil(max_sigma * 5) + 1)
        kernel_size = max (3, kernel_size)
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def _compute_spatial_kernel(self):
        sigma_sx, sigma_sy, sigma_sz, _ = self._get_activated_sigmas()
        coords = [torch.arange(-(self.kernel_size // 2), (self.kernel_size // 2) + 1, device=self.device) for _ in range(3)]
        coords = torch.meshgrid(*coords, indexing='ij')
        exponent = (
            -(coords[2].float() ** 2) / (2 * sigma_sx ** 2) +
            -(coords[1].float() ** 2) / (2 * sigma_sy ** 2) +
            -(coords[0].float() ** 2) / (2 * sigma_sz ** 2)
        )

        spatial_kernel = torch.exp(exponent)
        return spatial_kernel
    
    def _compute_range_kernel(self, center_values, neighbor_values):
        _, _, _, sigma_r = self._get_activated_sigmas()
        difference = (center_values - neighbor_values) ** 2
        range_kernel = torch.exp(-difference / (2 * sigma_r ** 2))
        return range_kernel
    
    def forward(self, x: torch.Tensor):
        if not x.dim() == 5:
            raise ValueError(f'Input must be a 5D tensor. Got shape: {x.shape}')
        x = x.float()

        self._update_kernel_size()

        B, C, D, H, W = x.shape
        output = torch.empty_like(x)
        spatial_kernel = self._compute_spatial_kernel().view(1, 1, 1, 1, 1, -1)
        
        for i in range(B):
            x_s = x[i:i+1]
            x_pad = torch.nn.functional.pad(x_s, [self.pad]*6, mode='reflect')
            patches = x_pad.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1).unfold(4, self.kernel_size, 1)
            patches = patches.contiguous().view(1, C, D, H, W, -1)
        
            range_kernel = self._compute_range_kernel(x_s.unsqueeze(-1), patches)
            kernel = spatial_kernel * range_kernel
            kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-8)
            output[i:i+1] = (patches * kernel).sum(dim=-1)
            del patches, kernel

        
        return output



class TemporalFilter(torch.nn.Module):
    def __init__(self, sigma_t, sigma_r, device):
        super(TemporalFilter, self).__init__()

        self.kernel_size = None
        self.pad = None

        self.sigma_t = torch.nn.Parameter(torch.tensor(inverse_softplus(sigma_t), dtype=torch.float32))
        self.sigma_r = torch.nn.Parameter(torch.tensor(inverse_softplus(sigma_r), dtype=torch.float32))
        
        self.temporal_activation = BoundedSoftplus(threshold=1.5)
        self.range_activation = BoundedSoftplus(threshold=6.0)
        self.device = device

    def _get_activated_sigmas(self):
        sigma_st = self.temporal_activation(self.sigma_t)
        sigma_r = self.range_activation(self.sigma_r)

        return sigma_st, sigma_r
    
    def _update_kernel_size(self):
        sigma_t, _ = self._get_activated_sigmas()
        kernel_size = int(2 * np.ceil(sigma_t.item() * 5) + 1)
        kernel_size = max(3, kernel_size)
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def _compute_temporal_kernel(self):
        sigma_t, _ = self._get_activated_sigmas()
        coords = torch.arange(-(self.kernel_size // 2), (self.kernel_size // 2) + 1, device=self.device)
        exponent = -(coords.float() ** 2) / (2 * sigma_t ** 2)
        temporal_kernel = torch.exp(exponent)

        return temporal_kernel
    
    def _compute_range_kernel(self, center_values, neighbors):
        _, sigma_r = self._get_activated_sigmas()
        difference = center_values - neighbors
        exponent = -(difference ** 2) / (2 * sigma_r ** 2)
        range_kernel = torch.exp(exponent)
        
        return range_kernel

    def forward(self, x: torch.Tensor):
        if not x.dim() == 6:
            raise ValueError(f'Input must be a 6D tensor. Got shape {x.shape}')

        x = x.float()

        self._update_kernel_size()

   
        B, C, T, D, H, W = x.shape
        x = x.permute(0, 3, 4, 5, 1, 2).view(B * D * H * W, C, T)
        x_pad = torch.nn.functional.pad(x, [self.pad]*2, mode='reflect')
        patches = x_pad.unfold(2, self.kernel_size, 1)
    
        temporal_kernel = self._compute_temporal_kernel().view(1, 1, 1, -1)
        range_kernel = self._compute_range_kernel(x.unsqueeze(-1), patches)
        kernel = temporal_kernel * range_kernel
        kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-8)
        filtered = (patches * kernel).sum(dim=-1).view(B, D, H, W, C, T).permute(0, 4, 5, 1, 2, 3)
        
        return filtered


class FilterBank(torch.nn.Module):
    def __init__(self, num_spatial, num_temporal):
        super(FilterBank, self).__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.spatial_kernels = torch.nn.ModuleList([SpatialFilter(
            sigma_sx=0.5,
            sigma_sy=0.5,
            sigma_sz=0.5,
            sigma_r=0.1,
            device=device,
        ) for _ in range(num_spatial)])

        self.temporal_kernels = torch.nn.ModuleList([TemporalFilter(
            sigma_t=0.5,
            sigma_r=0.1,
            device=device,
        ) for _ in range(num_temporal)])

    def forward(self, x):
        B, C, T, D, H, W = x.shape
        x = x.view(B*T, C, D, H, W)
        for stage in self.spatial_kernels:
            x = stage(x)

        x = x.view(B, C, T, D, H, W)
        for stage in self.temporal_kernels:
            x = stage(x)

        return x

