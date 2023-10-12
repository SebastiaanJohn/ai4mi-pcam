"""Utility functions for the project."""

from loguru import logger
from torch import nn
from torchvision import models

from src.config import settings
from src.engines.system import PCAMSystem


def get_model_at_version(model_name: str, version: str, model_type: str = "best-loss") -> tuple[str, str]:
    """Returns the path to the model at a given version and model type."""
    model_path = settings.experiments_dir / model_name

    if not model_path.exists():
        error = f"Model {model_name} does not exist. Please select an existing model or train a new one."
        raise ValueError(error)

    version_path = model_path / f"version_{version}"
    if not version_path.exists():
        error = f"Version {version} does not exist. Please select a valid version"
        raise ValueError(error)

    checkpoint_path = version_path / "checkpoints"
    found = False
    for file in checkpoint_path.iterdir():
        if model_type in file.name:
            model_path = checkpoint_path / file.name
            found = True
            break

    if not found:
        logger.warning(f"Could not find a model checkpoint with type {model_type} in {checkpoint_path}")

    logger.info(f"Loading model from path {model_path} and from version {version_path}")
    return version_path, model_path


def load_model_at_version(model_name: str, version: str, model_type: str = "best-loss") -> nn.Module:
    """Loads a model from a given version and model type."""
    _, model_path = get_model_at_version(model_name=model_name, version=version, model_type=model_type)

    model_loader = ModelLoader()
    model = model_loader.load_model(model_name=model_name)

    system = PCAMSystem.load_from_checkpoint(
        checkpoint_path=model_path,
        model=model,
    )
    logger.info(f"Successfully loaded {model_name} from {version}")

    return system

class ModelLoader:
    """Loads a model from the models module."""

    def __init__(self, num_classes: int = 2) -> None:
        """Initializes the ModelLoader class."""
        self.num_classes = num_classes

    def load_resnet34(self) -> nn.Module:
        """Loads a ResNet34 model."""
        model = models.resnet34()
        for param in model.parameters():
            param.requires_grad = False

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        return model

    def load_resnet18(self) -> nn.Module:
        """Loads a ResNet18 model."""
        model = models.resnet18()
        for param in model.parameters():
            param.requires_grad = False

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        return model

    def load_densenet121(self) -> nn.Module:
        """Loads a DenseNet121 model."""
        model = models.densenet121()
        for param in model.parameters():
            param.requires_grad = False

        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, self.num_classes)
        return model

    # Add more methods for other models...

    def load_model(self, model_name: str) -> nn.Module:
        """Loads a model from the models module."""
        model_loader_method = getattr(self, f"load_{model_name}", None)
        if model_loader_method and callable(model_loader_method):
            return model_loader_method()
        error_msg = f"Model {model_name} not supported."
        raise ValueError(error_msg)
