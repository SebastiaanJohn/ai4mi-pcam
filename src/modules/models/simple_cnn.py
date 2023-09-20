"""Simple CNN for debugging purposes."""

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """Simple CNN for debugging purposes."""

    def __init__(self) -> None:
        """Constructor for the DebugCNN class."""
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(nn.Linear(8 * 48 * 48, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
