from typing import Dict, List

from models.base import DenoisingModel
from models.cnn import CNN, MODEL_BUILDERS
from models.filterbank import FilterBank, STAGE_BUILDERS

def Model(stages: List[Dict]) -> DenoisingModel:
    """Builds the model family the stage types belong to"""
    types = {stage['type'] for stage in stages}
    if types <= STAGE_BUILDERS.keys():
        return FilterBank(stages)
    if types <= MODEL_BUILDERS.keys():
        return CNN(stages)

    raise KeyError(f'stage types {sorted(types)} belong to no single model family.   '
                   f'Filters: {sorted(STAGE_BUILDERS)}, CNNs: {sorted(MODEL_BUILDERS)}')