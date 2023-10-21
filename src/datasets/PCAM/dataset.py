"""Implementation of the PCAM dataset."""

from collections.abc import Callable
from pathlib import Path

import h5py
import torch
from PIL import Image
from PIL.Image import Image as ImageType
from torch.utils.data import Dataset


class PCAMDataset(Dataset):
    """PCAM dataset."""

    def __init__(
        self,
        data_file_path: Path,
        target_file_path: Path,
        transform: Callable | None = None,
    ) -> None:
        """PCAM Datset.

        Args:
            data_file_path (Path): Path to the data file.
            target_file_path (Path): Path to the target file.
            transform (Callable | None, optional): Transform to apply to the data.
                Defaults to None.
        """
        self._data_file_path = data_file_path
        self._target_file_path = target_file_path
        self._transform = transform

    def __len__(self) -> int:
        """Return the length of the dataset."""
        with h5py.File(self._data_file_path, "r") as f:
            return len(f["x"])

    def __getitem__(self, idx: int) -> tuple[ImageType, torch.Tensor]:
        """Return the item at the given index."""
        with h5py.File(self._data_file_path, "r") as f:
            image = Image.fromarray(f["x"][idx]).convert("RGB")

        with h5py.File(self._target_file_path, "r") as f:
            target = torch.tensor(f["y"][idx, 0, 0, 0], dtype=torch.float32)

        if self._transform:
            image = self._transform(image)

        return image, target


