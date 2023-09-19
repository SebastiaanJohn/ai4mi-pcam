"""Main interface for the PCAM dataset."""


import lightning.pytorch as pl
import torch
from PIL.Image import Image as ImageType
from torch import nn, optim


class PCAMClassifierModule(pl.LightningModule):
    """PyTorch Lightning module for PCAM dataset."""

    def __init__(self, model: nn.Module) -> None:
        """PyTorch Lightning module constructor."""
        super().__init__()
        self._model = model
        self._criterion = nn.CrossEntropyLoss()

    def training_step(
        self,
        batch: tuple[ImageType, torch.Tensor],
        batch_idx: int,
    ) -> float:
        """Return the loss for a training step."""
        sample, target = batch
        output = self._model(sample)
        loss = self._criterion(output, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self,
        batch: tuple[ImageType, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Return the loss for a validation step."""
        sample, target = batch
        output = self._model(sample)
        loss = self._criterion(output, target)
        self.log("val_loss", loss)

    def test_step(self, batch: tuple[ImageType, torch.Tensor], batch_idx: int) -> None:
        """Return the loss for a test step."""
        sample, target = batch
        output = self._model(sample)
        loss = self._criterion(output, target)
        self.log("test_loss", loss)

    def configure_optimizers(self) -> optim.Optimizer:
        """Configure the optimizer."""
        return optim.Adam(self.parameters(), lr=1e-3)
