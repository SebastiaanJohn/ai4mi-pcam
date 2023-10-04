"""Training script for the project."""

import sys
from argparse import ArgumentParser

import lightning.pytorch as pl
import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from loguru import logger

from src.config import settings
from src.datasets.PCAM import PCAMDataModule
from src.engines.system import PCAMSystem
from src.utils.callbacks import Callbacks
from src.utils.helpers import ModelLoader


def train(args) -> None:
    """Main training routine specific for this project."""
    logger.info("Starting training process...")

    # Setup Weights & Biases
    if args.wandb and not args.dev_run:
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
        data_dir=settings.raw_data_dir,
        lazy_loading=args.lazy_loading,
        crop_center=args.crop_center,
    )

    # Instantiate the model
    model_loader = ModelLoader()
    model = model_loader.load_model(args.model)
    system = PCAMSystem(model=model, compile_model=args.compile_model)

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

    wandb.finish()

if __name__ == "__main__":
    parser = ArgumentParser()

    # Dataset
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lazy_loading", type=bool, default=True)
    parser.add_argument("--crop_center", action="store_true")
    parser.add_argument("--train_size", type=float, default=None, help="Fraction of the training set to use.")
    parser.add_argument("--val_size", type=float, default=None, help="Fraction of the validation set to use.")

    # Model
    parser.add_argument("--model", type=str, default="simple_cnn")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early_stopping", action="store_true")

    # Logging
    parser.add_argument("--wandb", type=bool, default=False)

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
