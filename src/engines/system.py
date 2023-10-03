"""Main interface for the PCAM dataset."""


from typing import Any

import lightning.pytorch as pl
import torch
from torch import nn, optim
from torchmetrics import AUROC, Accuracy


class PCAMSystem(pl.LightningModule):
    """PyTorch Lightning module for the PCAM dataset."""

    def __init__(
        self, model: nn.Module, compile_model: bool, lr: float = 1e-3,
    ) -> None:
        """PyTorch Lightning module constructor."""
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        # Model
        self.model = model
        self.compile_model = compile_model

        # Hyperparameters
        self.lr = lr

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.train_auc = AUROC(task="binary")
        self.val_auc = AUROC(task="binary")
        self.test_auc = AUROC(task="binary")

    def _model_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[float, torch.Tensor, torch.Tensor]:
        """Performs a single step on a batch of data."""
        img, target = batch
        logits = self(img)
        loss = self.criterion(logits, target)
        preds = torch.argmax(logits, dim = 1)

        return loss, preds, target

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.model(img)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> float:
        """Return the loss for a training step."""
        loss, preds, targets = self._model_step(batch)

        # Update logs and metrics
        self.train_acc(preds, targets)
        self.train_auc(preds, targets)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/auc", self.train_auc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Return the loss for a validation step."""
        loss, preds, targets = self._model_step(batch)

        # Update logs and metrics
        self.val_acc(preds, targets)
        self.val_auc(preds, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Log the current learning rate at the end of the validation epoch."""
        self.best_acc = max(self.val_acc.compute(), self.best_acc if hasattr(self, "best_acc") else 0)
        self.best_auc = max(self.val_auc.compute(), self.best_auc if hasattr(self, "best_auc") else 0)
        self.log("val/best_acc", self.best_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/best_auc", self.best_auc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Return the loss for a test step."""
        loss, preds, targets = self._model_step(batch)

        # Update logs and metrics
        self.test_acc(preds, targets)
        self.test_auc(preds, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Compile the model if needed."""
        if self.hparams.compile_model and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure the optimizer."""
        return optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay if hasattr(self.hparams, "weight_decay") else 0,
        )

