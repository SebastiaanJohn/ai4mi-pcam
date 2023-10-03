"""Utility functions for the project."""

from torch import nn
from torchvision import models


class ModelLoader:
    """Loads a model from the models module."""

    def __init__(self, num_classes: int = 2) -> None:
        """Initializes the ModelLoader class."""
        self.num_classes = num_classes

    @classmethod
    def load_resnet34(cls) -> nn.Module:
        """Loads a ResNet34 model."""
        model = models.resnet34()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, cls.num_classes)
        return model

    @staticmethod
    def load_resnet18(cls) -> nn.Module:
        """Loads a ResNet18 model."""
        model = models.resnet18()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, cls.num_classes)
        return model

    # Add more methods for other models...

    @classmethod
    def load_model(cls, model_name: str) -> nn.Module:
        """Loads a model from the models module."""
        model_loader_method = getattr(cls, f"load_{model_name}", None)
        if model_loader_method and callable(model_loader_method):
            return model_loader_method()
        error_msg = f"Model {model_name} not supported."
        raise ValueError(error_msg)
