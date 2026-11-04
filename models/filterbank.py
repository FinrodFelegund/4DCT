from typing import Dict

import torch

from models.base import DenoisingModel

def _spatial3d(**kwargs):
    from bilateral_filter_layer_3d import BilateralFilter3d
    return BilateralFilter3d(**kwargs)

def _temporal4d(**kwargs):
    from bilateral_filter_layer_4d import BilateralFilter4d
    return BilateralFilter4d(**kwargs)

def _spt4d(**kwargs):
    from bilateral_filter_layer_spt import BilateralFilter4dspt
    return BilateralFilter4dspt(**kwargs)

def _centerweight4d(**kwargs):
    from bilateral_filter_layer_4d_weightedcenter import BilateralFilter4dspt
    return BilateralFilter4dspt(**kwargs)

def _learnable_spt4d(**kwargs):
    from bilateral_filter_layer_learnable_spt import BilateralFilter4dlearnablespt
    return BilateralFilter4dlearnablespt(**kwargs)

def _lut4d(**kwargs):
    from bilateral_filter_layer_4d_lut import BilateralFilter4DLUT
    return BilateralFilter4DLUT(**kwargs)

def _nd(**kwargs):
    from bilateral_filter_layer_nd import BilateralFilterNd
    return BilateralFilterNd(**kwargs)

def _ndlut(**kwargs):
    from bilateral_filter_layer_nd_lut import BilateralFilterNdLUT
    return BilateralFilterNdLUT(**kwargs)

def _ndembedding(**kwargs):
    from bilateral_filter_layer_nd_embedding import BilateralFilterNdEmbedding
    return BilateralFilterNdEmbedding(**kwargs)

STAGE_BUILDERS = {
    'spatial3d': _spatial3d,
    'temporal4d': _temporal4d,
    'spt4d': _spt4d,
    'centerweight4d': _centerweight4d,
    'nd': _nd,
    'ndlut': _ndlut,
    'learnable_spt4d': _learnable_spt4d,
    'lut4d': _lut4d,
    'ndembedding': _ndembedding,
}

RANGE_SUFFIXES = ('color_sigma', 'sigma_r')
CENTER_SUFFIXES = ('center_alpha',)
KERNEL_SUFFIXES = ('kernel',)
UNCONSTRAINED_SUFFIXES = ('raw_lut', 'lut')

class FilterBank(DenoisingModel):
    def __init__(self, stages):
        super().__init__()
        built, names = [], []
        for spec in stages:
            stage_type = spec['type']
            if stage_type not in STAGE_BUILDERS:
                raise KeyError(f'Unknown stage type {stage_type!r}. '
                               f'Known: {sorted(STAGE_BUILDERS)}')
            params = spec.get('params', {})
            for i in range(int(spec.get('count', 1))):
                built.append(STAGE_BUILDERS[stage_type](**params[i]))
                names.append(f'{stage_type}_{i + 1}')


        if not built:
            raise ValueError('Config declares no stages.')
        self.stages = torch.nn.ModuleList(built)
        self.stage_names = names
        self.supports_half_precision = all(getattr(s, 'supports_half_precision', False) for s in built)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x

    def parameter_groups(self, lr):
        groups = []
        for stage_name, stage in zip(self.stage_names, self.stages):
            if not hasattr(stage, 'parameter_groups'):
                raise TypeError(f'{stage_name}: {type(stage).__name__} has no parameter_groups(lr) function')
            for group in stage.parameter_groups(lr):
                groups.append({**group, 'name': f'{stage_name}/{group['name']}'})
        return groups

    def regularisation(self):
        penalties = [stage.lut_smoothness_penalty() for stage in self.stages
                     if hasattr(stage, 'lut_smoothness_penalty')]

        return sum(penalties) if penalties else None
    
    def scalar_parameters(self):
        out = {}
        for name, stage in zip(self.stage_names, self.stages):
            for param_name, param in stage.named_parameters():
                if param.numel() == 1:
                    out[f'{name}/{param_name}'] = param.item()

        return out



    @torch.no_grad()
    def clamp_(self, limits: Dict):
        sigma = (limits.get('sigma_min', 0.01), limits.get('sigma_max', 2.0))
        colour = (limits.get('range_min', 1e-4), limits.get('range_max', 2.0))
        centre = (limits.get('center_min', 0.0), limits.get('center_max', 10.0))
        kernel_min = limits.get('kernel_min', 0.0)

        for name, param in self.named_parameters():
            leaf = name.rsplit('.', 1)[-1]
            # log-parametrised values, LUTs and the range network are free
            if leaf.startswith('log_') or '.range_net.' in f'.{name}' or name.endswith(UNCONSTRAINED_SUFFIXES):
                continue
            if name.endswith(RANGE_SUFFIXES):
                param.clamp_(*colour)
            elif name.endswith(CENTER_SUFFIXES):
                param.clamp_(*centre)
            elif name.endswith(KERNEL_SUFFIXES):
                param.clamp_(min=kernel_min)
            else:
                param.clamp_(*sigma)

    def __repr__(self):
        return f'FilterBank({", ".join(self.stage_names)})\n' + '\n'.join(repr(s) for s in self.stages)


