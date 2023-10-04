"""Callbacks utility class."""

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

from src.config import settings


class Callbacks:
    """Utility class for callbacks."""

    def __init__(self) -> None:
        """Initializes the Callbacks class."""
        self.callbacks = []

    def add_tqdm_callback(self) -> None:
        """Adds a TQDM progress bar callback."""
        self.callbacks.append(TQDMProgressBar(refresh_rate=1))

    def add_checkpoint_callback(self, model_name: str, version: str) -> None:
        """Adds a checkpoint callback."""
        checkpoint_callback = ModelCheckpoint(
            dirpath=settings.experiments_dir / model_name / f"version_{version}" / "checkpoints",
            save_last=True,
            save_top_k=1,
            filename="best-loss-model-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
        )
        checkpoint_callback.CHECKPOINT_NAME_LAST = "last-model-{epoch:02d}-{val_loss:.2f}"
        self.callbacks.append(checkpoint_callback)

    def add_early_stopping_callback(self) -> None:
        """Adds an early stopping callback."""
        self.callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=True))

    def get_callbacks(self) -> list:
        """Returns the list of callbacks."""
        return self.callbacks
