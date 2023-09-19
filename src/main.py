"""Main module for the project."""

import sys
from argparse import ArgumentParser
from pathlib import Path

import lightning.pytorch as pl
from datasets.pcam import PCAMDataModule
from loguru import logger
from modules import PCAMLitModule
from modules.models import SimpleCNN


def main(args) -> None:
    """Main training routine specific for this project."""
    logger.info("Starting training process...")

    # Instantiate the data module
    data_module = PCAMDataModule(data_dir=args.data_dir, lazy_loading=True)

    # Instantiate the model
    model = SimpleCNN()
    lit_model = PCAMLitModule(model=model, compile_model=args.compile_model)

    # Instantiate the trainer
    trainer = pl.Trainer(fast_dev_run=args.dev_run)

    # Start the training process
    trainer.fit(lit_model, data_module)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=Path, default=Path("../data/raw"))
    parser.add_argument("--compile_model", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_scheduler", type=str, default=None)

    # Debugging
    parser.add_argument("--dev_run", type=bool, default=False)

    args = parser.parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG",
        colorize=True,
        format="<g>{time:YYYY-MM-DD HH:mm:ss}</g> | <lvl>{level}</lvl> | "
            "<c>{name}</c>:<c>{function}</c>:<c>{line}</c> - <lvl>{message}</lvl>",
    )

    main(args)
