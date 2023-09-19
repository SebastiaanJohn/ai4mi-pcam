"""Main module for the PCAM dataset."""

import logging
from pathlib import Path

import lightning.pytorch as pl
import torch
from dataset.datamodule import PCAMDataModule
from models.base_model import DebugCNN
from PIL.Image import Image as ImageType
from torch import nn, optim


class PCAMModule(pl.LightningModule):
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_dir = Path("/Users/sebastiaan/Developer/ai4mi-pcam/data")

    model = DebugCNN()
    pcam_module = PCAMModule(model)
    trainer = pl.Trainer(fast_dev_run=True, profiler="simple")
    data_module = PCAMDataModule(data_dir=data_dir, lazy_loading=True)
    trainer.fit(model=pcam_module, datamodule=data_module)
