"""Vision Transformer model with a custom classifier."""

import torch.nn as nn
from torchvision import models


def get_vit(version: str, freeze: bool = False, num_classes: int = 2) -> nn.Module:
    """Returns a Vision Transformer model with a custom classifier."""
    model = getattr(models, f"vit_{version}")()

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.heads.head.in_features
    model.heads = nn.Sequential(
        nn.Linear(num_features, num_classes),
    )

    return model
