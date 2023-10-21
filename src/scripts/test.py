"""Training script for the project."""

import sys
from argparse import ArgumentParser

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger
from torchvision import transforms

from src.config import settings
from src.datasets.pcam import PCAMDataModule
from src.engines.system import PCAMSystem
from src.utils.callbacks import Callbacks
from src.utils.helpers import get_model_at_version


def test(args) -> None:
    """Main test routine specific for this project."""
    logger.info("Starting test process...")

    # Setup logging
    pl_logger = TensorBoardLogger(
        save_dir=settings.root_dir,
        name=f"{settings.experiments_dir.name}/{args.model}",
        version=args.version,
    )

    # Load the callbacks
    callbacks = Callbacks()
    callbacks.add_tqdm_callback()

    # Instantiate the data module
    data_module = PCAMDataModule(
        data_dir=settings.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    )

    # Instantiate the model
    checkpoint_path = get_model_at_version(args.model, args.version)
    try:
        system = PCAMSystem.load_from_checkpoint(checkpoint_path)
    except RuntimeError:
        system = PCAMSystem.load_from_checkpoint(checkpoint_path, map_location="cpu")

    # Instantiate the trainer
    trainer = pl.Trainer(
        limit_val_batches=args.val_size if args.val_size else None,
        logger=pl_logger,
        callbacks=callbacks.get_callbacks(),
    )

    # Start the testing process
    trainer.test(model=system, datamodule=data_module)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Required
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)

    # Dataset
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_size", type=float, default=None, help="Fraction of the validation set to use.")

    # Debugging
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dev_run", action="store_true")

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

    test(args)
