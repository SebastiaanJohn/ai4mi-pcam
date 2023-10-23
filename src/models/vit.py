"""Vision Transformer model with a custom classifier."""

import torch.nn as nn
from torchvision import models


def get_vit(
    version: str,
    pretrained: bool = True,
    freeze: bool = False,
    num_classes: int = 1,
    ) -> nn.Module:
    """Returns a Vision Transformer model with a custom classifier."""
    weights = "IMAGENET1K_V1" if pretrained else None

    model = getattr(models, f"vit_{version}")(weights=weights)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.heads.head.in_features
    model.heads = nn.Sequential(
        nn.Linear(num_features, num_classes),
    )

    return model
