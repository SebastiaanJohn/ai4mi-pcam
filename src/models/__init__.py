"""__init__.py for models module."""

from typing import Any

import torch.nn as nn

from .densenet import get_densenet
from .resnet import get_resnet
from .vit import get_vit


MODEL_DICT = {
    "resnet": get_resnet,
    "densenet": get_densenet,
    "vit": get_vit,
}

def create_model(
    model_name: str,
    model_hparams: dict[str, Any],
    freeze: bool = False,
) -> nn.Module:
    """Creates a model from a given model name and model hyperparameters."""
    parts = model_name.split("_", 1)
    model_type = parts[0]
    version = parts[1]

    if model_type in MODEL_DICT:
        return MODEL_DICT[model_type](version=version, freeze=freeze, **model_hparams)

    error_msg = f"Model {model_name} not supported."
    raise ValueError(error_msg)
