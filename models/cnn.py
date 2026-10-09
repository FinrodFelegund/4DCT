import torch

def _redcnn2d(**kwargs):
    from models.RedCNN.red_cnn import REDCNN2D
    return REDCNN2D(**kwargs)

def _redcnn3d(**kwargs):
    from models.RedCNN.red_cnn import REDCNN3D
    return REDCNN3D(**kwargs)

MODEL_BUILDERS = {
    'redcnn2d': _redcnn2d,
    'redcnn3d': _redcnn3d,
}

class CNN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        model_type = config.get('type')
        if model_type not in MODEL_BUILDERS:
            raise KeyError(f'Unknown model type {model_type!r}   '
                           f'Known: {MODEL_BUILDERS}')

        params = config.get('params', {})
        network = MODEL_BUILDERS[model_type](**params)

        self.model = network
        self.model_type = model_type

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return x

    def parameter_groups(self, lr):
        if not hasattr(self.model, 'parameter_groups'):
            raise TypeError(f'{self.model_type}: {type(self.model).__name__} has no parameter_groups(lr) function')

        group = self.model.parameter_groups(lr)

        return group

    def scalar_parameters(self):
        return {}

    def regularisation(self):
        return None