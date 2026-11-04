import torch

from models.base import DenoisingModel


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

class CNN(DenoisingModel):
    def __init__(self, stages):
        super().__init__()
        if len(stages) != 1 or int(stages[0].get('count', 1)) != 1:
            raise ValueError(f'a CNN is a single stage with count 1, got {[s["type"] for s in stages]}')
        spec = stages[0]
        model_type = spec['type']
        if model_type not in MODEL_BUILDERS:
            raise KeyError(f'Unknown model type {model_type!r}. Known: {sorted(MODEL_BUILDERS)}')

        self.model = MODEL_BUILDERS[model_type](**spec.get('params', [{}])[0])
        self.model_type = model_type

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def parameter_groups(self, lr):
        return self.model.parameter_groups(lr)