"""ResNet model with a custom classifier."""

import torch.nn as nn
from torchvision import models


def get_resnet(
        version: str,
        pretrained: bool = True,
        freeze: bool = False,
        num_classes: int = 1,
    ) -> nn.Module:
    """Returns a ResNet model with a custom classifier."""
    if version == "50" and pretrained:
        weights = "IMAGENET1K_V2"
    elif version == "34" and pretrained:
        weights = "DEFAULT"
    else:
        weights = None

    model = getattr(models, f"resnet{version}")(weights=weights)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model

