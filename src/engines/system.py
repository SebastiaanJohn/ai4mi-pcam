"""Main interface for the PCAM dataset."""


from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn, optim
from torchmetrics import AUROC, Accuracy

from src.models import create_model


class PCAMSystem(pl.LightningModule):
    """PyTorch Lightning module for the PCAM dataset."""

    def __init__(
        self,
        model_name: str,
        model_hparams: dict[str, Any],
        optimizer_name: str,
        optimizer_hparams: dict[str, Any],
        compile_model: bool = False,
        freeze: bool = False,
    ) -> None:
        """PyTorch Lightning module constructor."""
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = create_model(model_name, model_hparams, freeze)
        self.compile_model = compile_model

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

        self.train_auc = AUROC(task="binary")
        self.val_auc = AUROC(task="binary")
        self.test_auc = AUROC(task="binary")

    def model_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[float, torch.Tensor, torch.Tensor]:
        """Performs a single step on a batch of data."""
        img, targets = batch
        output = self(img)

        # The following code is needed to support Inception V3
        if isinstance(output, tuple) and hasattr(output, "logits"):
            logits = output.logits.squeeze()
        else:
            logits = output.squeeze()

        loss = self.criterion(logits, targets)

        return loss, logits, targets

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.model(img)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> float:
        """Return the loss for a training step."""
        loss, preds, targets = self.model_step(batch)

        # Update logs and metrics
        self.train_acc(preds, targets)
        self.train_auc(preds, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Return the loss for a validation step."""
        loss, preds, targets = self.model_step(batch)

        # Update logs and metrics
        self.val_acc(preds, targets)
        self.val_auc(preds, targets)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Return the loss for a test step."""
        loss, preds, targets = self.model_step(batch)

        # Update logs and metrics
        self.test_acc(preds, targets)
        self.test_auc(preds, targets)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """Return the predictions for a batch."""
        img, _ = batch
        logits = self(img)

        return F.sigmoid(logits)

    def setup(self, stage: str) -> None:
        """Compile the model if needed."""
        if self.hparams.compile_model and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure the optimizer and the learning rate scheduler."""
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            error_msg = f'Unknown optimizer: "{self.hparams.optimizer_name}"'
            raise AssertionError(error_msg)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

