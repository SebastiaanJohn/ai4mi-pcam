"""Implementation of the PCAM dataset."""

from collections.abc import Callable
from pathlib import Path

import h5py
import torch
from PIL import Image
from PIL.Image import Image as ImageType
from torch.utils.data import Dataset
from torchvision import transforms


class PCAMDataset(Dataset):
    """PCAM dataset."""

    def __init__(
        self,
        data_file_path: Path,
        target_file_path: Path,
        transform: Callable | None = None,
        lazy_loading: bool = False,
    ) -> None:
        """PCAM dataset constructor.

        Args:
            data_file_path (Path): Path to the data file.
            target_file_path (Path): Path to the target file.
            transform (Callable | None, optional): Transform to apply to the data.
            Defaults to None.
            lazy_loading (bool, optional): Whether to load the data lazily.
            Defaults to False.
        """
        self._data_file_path = data_file_path
        self._target_file_path = target_file_path
        self._transform = transform
        self._lazy_loading = lazy_loading

        with h5py.File(self._data_file_path, "r") as f:
            self._data_length = len(f["x"])
            if not self._lazy_loading:
                self._data = f["x"][:]

        with h5py.File(self._target_file_path, "r") as f:
            _target_length = len(f["y"])
            if not self._lazy_loading:
                self._target = torch.tensor(f["y"][:], dtype=torch.long).squeeze()

        if self._data_length != _target_length:
            msg = "Mismatch between data and target lengths."
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self._data) if not self._lazy_loading else self._data_length

    def __getitem__(self, idx: int) -> tuple[ImageType, torch.Tensor]:
        """Return the item at the given index."""
        if self._lazy_loading:
            with h5py.File(self._data_file_path, "r") as f:
                sample = Image.fromarray(f["x"][idx])

            with h5py.File(self._target_file_path, "r") as f:
                target = torch.tensor(f["y"][idx], dtype=torch.long).squeeze()
        else:
            sample = Image.fromarray(self._data[idx])
            target = self._target[idx]

        sample = self._transform(sample) if self._transform else transforms.ToTensor()(sample)

        return sample, target
