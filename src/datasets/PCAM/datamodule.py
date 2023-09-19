"""PyTorch Lightning DataModule for PCAM dataset."""

from pathlib import Path

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from .dataset import PCAMDataset


class PCAMDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for PCAM dataset."""

    def __init__(self,
        data_dir: Path,
        batch_size: int = 32,
        num_workers: int = 8,
        lazy_loading: bool = False,
    ) -> None:
        """Initialize the PCAMDataModule."""
        super().__init__()
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._lazy_loading = lazy_loading
        self._transform = None

        self.num_classes = 2

    def setup(self, stage: str) -> None:
        """Set up the data."""
        if stage == "fit":
            self._train_dataset = PCAMDataset(
                data_file_path=self._data_dir
                / "camelyonpatch_level_2_split_train_x.h5",
                target_file_path=self._data_dir
                / "camelyonpatch_level_2_split_train_y.h5",
                transform=self._transform,
                lazy_loading=self._lazy_loading,
            )
            self._val_dataset = PCAMDataset(
                data_file_path=self._data_dir
                / "camelyonpatch_level_2_split_valid_x.h5",
                target_file_path=self._data_dir
                / "camelyonpatch_level_2_split_valid_y.h5",
                transform=self._transform,
                lazy_loading=self._lazy_loading,
            )

        if stage == "test":
            self._test_dataset = PCAMDataset(
                data_file_path=self._data_dir / "camelyonpatch_level_2_split_test_x.h5",
                target_file_path=self._data_dir
                / "camelyonpatch_level_2_split_test_y.h5",
                transform=self._transform,
                lazy_loading=self._lazy_loading,
            )

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )
