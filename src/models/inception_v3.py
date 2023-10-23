"""Inception V3 model with a custom classifier."""

import torch.nn as nn
from torchvision import models


def get_inception_v3(
        version: str | None = None,
        pretrained: bool = True,
        freeze: bool = False,
        num_classes: int = 1,
    ) -> nn.Module:
    """Returns a inception_v3 model with a custom classifier."""
    weights = "DEFAULT" if pretrained else None

    model = models.inception_v3(weights=weights)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model
