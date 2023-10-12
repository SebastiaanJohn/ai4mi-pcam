"""__init__.py for models module."""

from typing import Any

import torch.nn as nn

from .densenet import get_densenet
from .resnet import get_resnet


MODEL_DICT = {
    "resnet": get_resnet,
    "densenet": get_densenet,
}

def create_model(model_name: str, model_hparams: dict[str, Any]) -> nn.Module:
    """Creates a model from a given model name and model hyperparameters."""
    model_type, version = model_name.split("_")
    if model_type in MODEL_DICT:
        return MODEL_DICT[model_type](version=version, **model_hparams)
    error_msg = f"Model {model_name} not supported."
    raise ValueError(error_msg)
