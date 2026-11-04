from abc import ABC, abstractmethod
from typing import Dict, List

import torch

class DenoisingModel(torch.nn.Module, ABC):

    supports_half_precision: bool = True

    @abstractmethod
    def parameter_groups(self, lr:Dict[str, float]) -> List[Dict]:
        """Optimizer groups"""

    def optimizer_groups(self, lr: Dict[str, float]) -> List[Dict]:
        groups = [g for g in self.parameter_groups(lr) if g['params']]

        grouped = [id(p) for g in groups for p in g['params']]
        if len(grouped) != len(set(grouped)):
            raise ValueError('a parameter appears in more than one optimizer group')
        missing = [n for n, p in self.named_parameters() if p.requires_grad and id(p) not in set(grouped)]
        if missing:
            raise ValueError(f'trainable parameters without an optimizer group: {missing}')
        return groups

    def regularisation(self):
        """Model specific regularisation"""
        return None

    def scalar_parameters(self) -> Dict[str, float]:
        """Model specific scalar parameters like sigma values"""
        return {}

    @torch.no_grad()
    def clamp_(self, limits: Dict):
        """Clamps constrained parameters back into their valid range after an optiization step"""

    def forward_volume(self, x: torch.Tensor):
        return self(x)