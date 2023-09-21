"""Main module for the project."""
import sys
from argparse import ArgumentParser
from pathlib import Path

import lightning.pytorch as pl
import wandb
from datasets.pcam import PCAMDataModule
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from loguru import logger
from modules import PCAMLitModule
from pytorch_lightning.loggers import WandbLogger
from utils import load_model


def main(args) -> None:
    """Main training routine specific for this project."""
    logger.info("Starting training process...")

    # Setup Weights & Biases
    if args.wandb and not args.dev_run:
        wandb_logger = (
            WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                save_dir=Path(__file__).resolve().parent.parent / args.wandb_dir,
            )
        )
        wandb_logger.experiment.config["batch_size"] = args.batch_size
    else:
        wandb_logger = None

    # Instantiate the data module
    data_module = PCAMDataModule(
        data_dir=args.data_dir,
        lazy_loading=args.lazy_loading,
        crop_center=args.crop_center,
    )

    # Instantiate the model
    model = load_model(args.model, input_size=32 if args.crop_center else 96)
    lit_model = PCAMLitModule(model=model, compile_model=args.compile_model)

    callbacks = []
    if args.early_stopping:
        early_stopping = EarlyStopping(monitor="val/loss", mode="min")
        callbacks.append(early_stopping)

    # Instantiate the trainer
    trainer = pl.Trainer(
        fast_dev_run=args.dev_run,
        limit_train_batches=args.train_size,
        limit_val_batches=args.val_size,
        logger=wandb_logger,
        max_epochs=args.epochs,
        callbacks=callbacks,
    )

    # Start the training process
    trainer.fit(lit_model, data_module)

    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()

    # Dataset
    parser.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lazy_loading", type=bool, default=True)
    parser.add_argument("--crop_center", action="store_true")
    parser.add_argument("--train_size", type=float, default=1, help="Fraction of the training set to use.")
    parser.add_argument("--val_size", type=float, default=1, help="Fraction of the validation set to use.")

    # Model
    parser.add_argument("--model", type=str, default="simple_cnn")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_scheduler", type=str, default=None)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early_stopping", action="store_true")

    # Logging
    parser.add_argument("--wandb", type=bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="pcam-classification")
    parser.add_argument("--wandb_entity", type=str, default="ai4mi")
    parser.add_argument("--wandb_dir", type=str, default="experiments")

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

    main(args)
