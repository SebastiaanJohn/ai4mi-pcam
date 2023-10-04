"""Utility functions for the project."""

from torch import nn
from torchvision import models


class ModelLoader:
    """Loads a model from the models module."""

    def __init__(self, num_classes: int = 2) -> None:
        """Initializes the ModelLoader class."""
        self.num_classes = num_classes

    def load_resnet34(self) -> nn.Module:
        """Loads a ResNet34 model."""
        model = models.resnet34()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        return model

    def load_resnet18(self) -> nn.Module:
        """Loads a ResNet18 model."""
        model = models.resnet18()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        return model

    # Add more methods for other models...

    def load_model(self, model_name: str) -> nn.Module:
        """Loads a model from the models module."""
        model_loader_method = getattr(self, f"load_{model_name}", None)
        if model_loader_method and callable(model_loader_method):
            return model_loader_method()
        error_msg = f"Model {model_name} not supported."
        raise ValueError(error_msg)
