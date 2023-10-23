"""DenseNet model with a custom classifier."""

import torch.nn as nn
from torchvision import models


def get_densenet(
        version: str,
        pretrained: bool = True,
        freeze: bool = False,
        num_classes: int = 1,
    ) -> nn.Module:
    """Returns a DenseNet model with a custom classifier."""
    weights = "IMAGENET1K_V1" if pretrained and version == "121" else None

    model = getattr(models, f"densenet{version}")(weights=weights)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)

    return model
