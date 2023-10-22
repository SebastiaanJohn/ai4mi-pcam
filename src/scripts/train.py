"""Training script for the project."""

import sys
from argparse import ArgumentParser

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from loguru import logger
from torchvision import transforms

from src.config import settings
from src.datasets.pcam import PCAMDataModule
from src.engines.system import PCAMSystem
from src.utils.callbacks import Callbacks


def train(args) -> None:
    """Main training routine specific for this project."""
    logger.info("Starting training process...")

    # Setup Weights & Biases
    if args.wandb:
        pl_logger = WandbLogger(
            save_dir=settings.root_dir,
            name=f"{settings.experiments_dir.name}/{args.model}",
            project=settings.wandb_project,
            entity=settings.wandb_entity,
        )
    else:
        pl_logger = TensorBoardLogger(
            save_dir=settings.root_dir,
            name=f"{settings.experiments_dir.name}/{args.model}",
        )

    # Load the callbacks
    callbacks = Callbacks()
    callbacks.add_tqdm_callback()
    callbacks.add_checkpoint_callback(args.model, pl_logger.version)
    callbacks.add_early_stopping_callback()

    # Instantiate the data module
    data_module = PCAMDataModule(
        data_dir=settings.processed_data_dir / "pcam",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transforms=transforms.Compose([
            # transforms.Resize((224, 224)), uncomment this line for ViT_b_16
            transforms.Resize((299, 299)),  # uncomment this line for Inception V3
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    )

    # Parameters
    model_hparams = {
        "num_classes": data_module.num_classes,
    }
    optimizer_hparams = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }

    # Instantiate the model
    system = PCAMSystem(
        model_name=args.model,
        model_hparams=model_hparams,
        optimizer_name=args.optimizer,
        optimizer_hparams=optimizer_hparams,
        compile_model=args.compile_model,
        freeze=args.freeze,
    )

    # Instantiate the trainer
    trainer = pl.Trainer(
        default_root_dir=settings.experiments_dir,
        fast_dev_run=args.dev_run,
        limit_train_batches=args.train_size if args.train_size else None,
        limit_val_batches=args.val_size if args.val_size else None,
        logger=pl_logger,
        max_epochs=args.epochs,
        callbacks=callbacks.get_callbacks(),
    )

    # Start the training process
    trainer.fit(model=system, datamodule=data_module)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Dataset
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--train_size", type=float, default=None, help="Fraction of the training set to use.")
    parser.add_argument("--val_size", type=float, default=None, help="Fraction of the validation set to use.")

    # Model
    parser.add_argument("--model", type=str, default="resnet_34")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze", action="store_true")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--optimizer", type=str, default="Adam")

    # Logging
    parser.add_argument("--wandb", action="store_true")

    # Debugging
    parser.add_argument("--dev_run", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # Configure the logger
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if args.debug else "INFO",
        colorize=True,
        format=(
            "<g>{time:YYYY-MM-DD HH:mm:ss}</g> | <lvl>{level}</lvl> | "
            "<c>{name}</c>:<c>{function}</c>:<c>{line}</c> - <lvl>{message}</lvl>"
        ),
    )

    train(args)
