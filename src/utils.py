"""Utility functions for the project."""

from modules.models import SimpleCNN
from torch import nn
from torchvision import models


def load_model(model_name: str, input_size: int) -> nn.Module:
    """Loads a model from the models module."""
    match model_name:
        case "simple_cnn":
            return SimpleCNN(input_size=input_size)
        case "resnet50":
            model = models.resnet50()

            # Replace the last layer
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 2)

            return model
        case _:
            error_msg = f"Model {model_name} not supported."
            raise ValueError(error_msg)
