"""__init__.py for models module."""

from typing import Any

import torch.nn as nn

from .alexnet import get_alexnet
from .densenet import get_densenet
from .inception_v3 import get_inception_v3
from .resnet import get_resnet
from .vgg11 import get_vgg11
from .vit import get_vit


MODEL_DICT = {
    "resnet": get_resnet,
    "densenet": get_densenet,
    "vit": get_vit,
    "alexnet": get_alexnet,
    "inception": get_inception_v3,
    "vgg": get_vgg11,
}

def create_model(
    model_name: str,
    model_hparams: dict[str, Any],
    freeze: bool = False,
) -> nn.Module:
    """Creates a model from a given model name and model hyperparameters."""
    parts = model_name.split("_", 1)
    model_type = parts[0]
    version = None if len(parts) == 1 else parts[1]

    if model_type in MODEL_DICT:
        return MODEL_DICT[model_type](version=version, freeze=freeze, **model_hparams)

    error_msg = f"Model {model_name} not supported."
    raise ValueError(error_msg)
