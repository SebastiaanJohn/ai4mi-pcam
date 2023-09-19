"""Main interface for the PCAM dataset."""


import importlib
import inspect
from typing import Any

import lightning.pytorch as pl
import torch
import torch.optim.lr_scheduler as lrs
from PIL.Image import Image
from torch import nn
from torchmetrics import Accuracy


class PCAMLitModule(pl.LightningModule):
    """PyTorch Lightning module for PCAM dataset."""

    def __init__(self, model: nn.Module, compile_model: bool) -> None:
        """PyTorch Lightning module constructor."""
        super().__init__()
        self.save_hyperparameters()
        self._load_model()
        self.compile_model = compile_model
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = Accuracy(task="binary", num_classes=2)
        self.val_acc = Accuracy(task="binary", num_classes=2)
        self.test_acc = Accuracy(task="binary", num_classes=2)

    def _load_model(self) -> None:
        """Load the model."""
        model_name = self.hparams.model
        camel_name = "".join(word.capitalize() for word in model_name.split("_"))
        try:
            model = getattr(importlib.import_module(f".{model_name}", package=__package__), camel_name)
        except (ImportError, AttributeError) as e:
            error_msg = f"Failed to import the model class '{camel_name}' from the module '.{model_name}'. " \
                        f"Please make sure the module and class names are correct."
            raise ValueError(error_msg) from e
        self.model = self._initialize_model(model)

    def _initialize_model(self, model: type, **kwargs) -> nn.Module:
        """Initialize a model with parameters from self.hparams dictionary."""
        class_args = inspect.getfullargspec(model.__init__).args[1:]
        model_args = {arg: getattr(self.hparams, arg) for arg in class_args if arg in self.hparams}
        model_args.update(kwargs)
        return model(**model_args)

    def _model_step(self, batch: tuple[Image, torch.Tensor]) -> float:
        """Performs a single step on a batch of data."""
        img, target = batch
        logits = self(img)
        loss = self.criterion(logits, target)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, target

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.model(img)

    def training_step(self, batch: tuple[Image, torch.Tensor], batch_idx: int) -> float:
        """Return the loss for a training step."""
        loss, preds, targets = self._model_step(batch)

        # Update logs and metrics
        self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[Image, torch.Tensor], batch_idx: int) -> None:
        """Return the loss for a validation step."""
        loss, preds, targets = self._model_step(batch)

        # Update logs and metrics
        self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Log the current learning rate at the end of the validation epoch."""
        # TODO: Get best accuracy and log it

    def test_step(self, batch: tuple[Image, torch.Tensor], batch_idx: int) -> None:
        """Return the loss for a test step."""
        loss, preds, targets = self._model_step(batch)

        # Update logs and metrics
        self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Compile the model if needed."""
        if self.hparams.compile_model and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure the optimizer."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
            if hasattr(self.hparams, "weight_decay")
            else 0,
        )

        if self.hparams.lr_scheduler is None:
            return optimizer

        if self.hparams.lr_scheduler == "step":
            scheduler = lrs.StepLR(
                optimizer,
                step_size=self.hparams.lr_decay_steps,
                gamma=self.hparams.lr_decay_rate,
            )
        else:
            error_msg = "Invalid lr_scheduler type!"
            raise ValueError(error_msg)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

