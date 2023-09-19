"""Base model class for all models."""

import torch
import torch.nn as nn


class DebugCNN(nn.Module):
    """Simple CNN for debugging purposes."""

    def __init__(self) -> None:
        """Constructor for the DebugCNN class."""
        super().__init__()

        # Convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1,
        )

        # Output layer
        self.fc1 = nn.Linear(8 * 48 * 48, 2)

        # Activation and pooling
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        # Conv block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Flatten and output
        x = x.view(x.size(0), -1)

        return self.fc1(x)
