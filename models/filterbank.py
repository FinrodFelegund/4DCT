import torch

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
    from bilater_filter_layer_nd_lut import BilaterFilterNdLUT
    return BilaterFilterNdLUT(**kwargs)

STAGE_BUILDERS = {
    'spatial3d': _spatial3d,
    'temporal4d': _temporal4d,
    'spt4d': _spt4d,
    'centerweight4d': _centerweight4d,
    'nd': _nd,
    'ndlut': _ndlut,
    'learnable_spt4d': _learnable_spt4d,
    'lut4d': _lut4d,
}

class FilterBank(torch.nn.Module):
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

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x

    def scalar_parameters(self):
        out = {}
        for name, stage in zip(self.stage_names, self.stages):
            for param_name, param in stage.named_parameters():
                if param.numel() == 1:
                    out[f'{name}/{param_name}'] = param.item()

        return out

    def regularisation(self):
        total = None
        for stage in self.stages:
            penalty_fn = getattr(stage, 'lut_smoothness_penalty', None)
            if penalty_fn is None:
                continue
            
            penalty = penalty_fn()
            total = penalty if total is None else total + penalty

        return total

    def parameter_groups(self, lr):
        groups = []
        for stage_name, stage in zip(self.stage_names, self.stages):
            if not hasattr(stage, 'parameter_groups'):
                raise TypeError(f'{stage_name}: {type(stage).__name__} has no parameter_groups(lr) function')
            for group in stage.parameter_groups(lr):
                if group['params']:
                    groups.append({**group, 'name': f'{stage_name}/{group['name']}'})

        grouped = [id(p) for g in groups for p in g['params']]
        if len(grouped) != len(set(grouped)):
            raise ValueError('a parameter appears in more than one optimizer group')
        missing = [n for n, p in self.named_parameters() if p.requires_grad and id(p) not in set(grouped)]
        if missing:
            raise ValueError(f'trainable parameters without an optimizer group: {missing}')

        return groups

    def __repr__(self):
        for stage in self.stages:
            print(stage)
        return f'Filterbank({", ".join(self.stage_names)})'


