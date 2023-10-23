"""AlexNet model with a custom classifier."""

import torch.nn as nn
from torchvision import models


def get_alexnet(
        version: str | None = None,
        pretrained: bool = True,
        freeze: bool = False,
        num_classes: int = 1,
    ) -> nn.Module:
    """Returns a alexnet model with a custom classifier."""
    weights = "DEFAULT" if pretrained else None

    model = models.alexnet(weights=weights)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)

    return model
