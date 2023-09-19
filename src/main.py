"""Main module for the project."""

from argparse import ArgumentParser
from pathlib import Path

import lightning.pytorch as pl
from datasets.pcam import PCAMDataModule
from modules import PCAMLitModule
from modules.models import SimpleCNN


def main(args) -> None:
    """Main training routine specific for this project."""
    # Instantiate the data module
    data_module = PCAMDataModule(data_dir=args.data_dir, lazy_loading=True)

    # Instantiate the model
    model = SimpleCNN()
    lit_model = PCAMLitModule(model=model, compile_model=args.compile_model)

    # Instantiate the trainer
    trainer = pl.Trainer(fast_dev_run=100)

    # Start the training process
    trainer.fit(lit_model, data_module)

if __name__ == "__main__":
    parser = ArgumentParser()

    # Add program level args
    parser.add_argument("--data_dir", type=Path, default=Path("../data/raw"))
    parser.add_argument("--compile_model", type=bool, default=False)

    args = parser.parse_args()

    main(args)
