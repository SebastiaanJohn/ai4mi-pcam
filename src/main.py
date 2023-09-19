"""Main module for the PCAM dataset."""


import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI

from .datasets.pcam import PCAMDataModule
from .modules.interface import PCAMClassifierModule


def cli_main() -> None:
    """Main method for the PCAM dataset."""
    pl.seed_everything(42)
    LightningCLI(PCAMClassifierModule, PCAMDataModule)


if __name__ == "__main__":
    cli_main()
