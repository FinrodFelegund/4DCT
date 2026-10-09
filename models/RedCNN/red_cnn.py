"""
Author: Daniel Pietsch
Contact: daniel.pietsch@studium.uni-hamburg.de

Implementation of Red CNN by Chen et al., incorporating time dimension
"""

from typing import Tuple

import torch
import torch.nn.functional as F

class REDCNN3D(torch.nn.Module):
    def __init__(self, phases: int = 10, channels: int = 96, kernel: Tuple = (5, 5, 3)):
        super().__init__()

        self.convs = torch.nn.ModuleList([torch.nn.Conv3d(
            in_channels=phases if i == 0 else channels,
            out_channels=channels,
            kernel_size=kernel,
            stride=1,
            padding=0
        ) for i in range(5)])

        self.deconvs = torch.nn.ModuleList([torch.nn.ConvTranspose3d(
            in_channels=channels,
            out_channels=phases if i == 4 else channels,
            kernel_size=kernel,
            stride=1,
            padding=0
        ) for i in range(5)])

        self.init_weights()


    def init_weights(self):
        """
        Initialisation as described in Chen et al.
        """
        for layer in [*self.convs, *self.deconvs]:
            torch.nn.init.normal_(layer.weight, 0.0, 0.01)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor):
        """
        No final relu, unlike the paper the paper, since the voxels in target and noisy can be negative by design of the noise projection
        """
        assert x.dim() == 6, 'Input shape must be [B, C, T, D, H, W]'
        assert x.shape[1] == 1, 'Currently only one channel images are supported'

        B, C, T, D, H, W = x.shape
        x = x.permute(0, 1, 2, 4, 5, 3).reshape((B, C * T, H, W, D)).contiguous()

        r1 = x
        h = F.relu(self.convs[0](x))
        h = F.relu(self.convs[1](h))
        r2 = h
        h = F.relu(self.convs[2](h))
        h = F.relu(self.convs[3](h))
        r3 = h
        h = F.relu(self.convs[4](h))

        h = self.deconvs[0](h) + r3
        h = self.deconvs[1](F.relu(h))
        h = self.deconvs[2](F.relu(h)) + r2
        h = self.deconvs[3](F.relu(h))
        h = self.deconvs[4](F.relu(h)) + r1

        h = h.reshape(B, C, T, H, W, D).permute(0, 1, 2, 5, 3, 4).contiguous()
        return h

    def parameter_groups(self, lr):
        return [
            {'name': 'network', 'params': list(self.parameters()), 'lr': lr['network'], 'weight_decay': 0.0}
        ]

    def scalar_parameters(self):
        return {}

    def regularisation(self):
        return None


class REDCNN2D(torch.nn.Module):
    def __init__(self, channels: int = 96, kernel: Tuple = (5, 5)):
        super().__init__()
        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(1 if i == 0 else channels, channels, kernel) for i in range(5)])
        self.deconvs = torch.nn.ModuleList([torch.nn.ConvTranspose2d(channels, 1 if i == 4 else channels, kernel) for i in range(5)])

        self.init_weights()

    def init_weights(self):
        """
        Initialisation as described in Chen et al.
        """
        for layer in [*self.convs, *self.deconvs]:
            torch.nn.init.normal_(layer.weight, 0, 0.01)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor):
        """No final relu same reason as above"""
        assert x.dim() == 6, 'Input shape must be [B, C, T, D, H, W]'
        assert x.shape[1] == 1, 'Currently only one channel images are supported'

        B, C, T, D, H, W = x.shape
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B * T * D, C, H, W).contiguous()

        r1 = x
        h = F.relu(self.convs[0](x))
        h = F.relu(self.convs[1](h))
        r2 = h
        h = F.relu(self.convs[2](h))
        h = F.relu(self.convs[3](h))
        r3 = h
        h = F.relu(self.convs[4](h))

        h = self.deconvs[0](h) + r3
        h = self.deconvs[1](F.relu(h))
        h = self.deconvs[2](F.relu(h)) + r2
        h = self.deconvs[3](F.relu(h))
        h = self.deconvs[4](F.relu(h)) + r1

        h = h.reshape(B, T, D, C, H, W).permute(0, 3, 1, 2, 4, 5).contiguous()
        return h

    def parameter_groups(self, lr):
        return [
            {'name': 'network', 'params': list(self.parameters()), 'lr': lr['network'], 'weight_decay': 0.0}
        ]

    def scalar_parameters(self):
        return {}

    def regularisation(self):
        return None
