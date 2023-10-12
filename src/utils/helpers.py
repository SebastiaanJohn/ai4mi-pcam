"""Utility functions for the project."""


from loguru import logger

from src.config import settings


def get_model_at_version(model_name: str, version: str, model_type: str = "best-loss") -> str:
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

    logger.info(f"Loading model from path {model_path}")

    return model_path



