"""DenseNet model with a custom classifier."""

import torch.nn as nn
from torchvision import models


def get_densenet(version: str, freeze: bool = False, num_classes: int = 2) -> nn.Module:
    """Returns a DenseNet model with a custom classifier."""
    model = getattr(models, f"densenet{version}")()

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)

    return model
