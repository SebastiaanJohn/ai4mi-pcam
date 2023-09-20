"""Main module for the project."""
import sys
from argparse import ArgumentParser
from pathlib import Path

import lightning.pytorch as pl
import wandb
from datasets.pcam import PCAMDataModule
from loguru import logger
from modules import PCAMLitModule
from modules.models import SimpleCNN
from pytorch_lightning.loggers import WandbLogger


def main(args) -> None:
    """Main training routine specific for this project."""
    logger.info("Starting training process...")

    # Setup Weights & Biases
    wandb_logger = (
        WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            save_dir=Path(__file__).resolve().parent.parent / args.wandb_dir,
        )
        if args.wandb
        else None
    )

    wandb_logger.experiment.config["batch_size"] = args.batch_size

    # Instantiate the data module
    data_module = PCAMDataModule(data_dir=args.data_dir, lazy_loading=True)

    # Instantiate the model
    model = SimpleCNN()
    lit_model = PCAMLitModule(model=model, compile_model=args.compile_model)

    # Instantiate the trainer
    trainer = pl.Trainer(fast_dev_run=args.dev_run, logger=wandb_logger, max_epochs=args.epochs)

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

    # Model
    parser.add_argument("--compile_model", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_scheduler", type=str, default=None)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early_stopping", type=bool, default=False)

    # Logging
    parser.add_argument("--wandb", type=bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="pcam-classification")
    parser.add_argument("--wandb_entity", type=str, default="ai4mi")
    parser.add_argument("--wandb_dir", type=str, default="experiments")

    # Debugging
    parser.add_argument("--dev_run", type=bool, default=False, help="Whether to run a development run.")
    parser.add_argument("--debug", type=bool, default=False, help="Whether to run in debug mode.")

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
