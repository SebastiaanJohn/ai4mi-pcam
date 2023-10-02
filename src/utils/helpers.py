"""Utility functions for the project."""

from torch import nn
from torchvision import models


NUM_CLASSES = 2

def load_model(model_name: str, input_size: int) -> nn.Module:
    """Loads a model from the models module."""
    match model_name:
        case "resnet50":
            model = models.resnet50()

            # Replace the last layer
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, NUM_CLASSES)

            return model
        case "resnet18":
            model = models.resnet18(weights="DEFAULT")

            # Freeze all layers
            for param in model.parameters():
                param.requires_grad = False

            # Replace the last layer
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, NUM_CLASSES)

            return model

        case "resnet34":
            model = models.resnet34(weights="DEFAULT")

            # Freeze all layers
            for param in model.parameters():
                param.requires_grad = False

            # Replace the last layer
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, NUM_CLASSES)

            return model
        case _:
            error_msg = f"Model {model_name} not supported."
            raise ValueError(error_msg)
