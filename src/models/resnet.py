"""ResNet model with a custom classifier."""

import torch.nn as nn
from torchvision import models


def get_resnet(version: str, freeze: bool = False, num_classes: int = 2) -> nn.Module:
    """Returns a ResNet model with a custom classifier."""
    model = getattr(models, f"resnet{version}")()

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model

