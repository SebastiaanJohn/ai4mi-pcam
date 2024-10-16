"""Helper functions for XAI."""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


def predict_labels(imgs_preprocessed: torch.Tensor, model: nn.Module, requires_grad: bool = True) -> torch.Tensor:
    """Predict the an image's label using the given model.

    Note: If the model has only one output neuron, the output tensor will contain two columns. The first column
    contains the probability of the image belonging to the positive class, and the second column is the probability of
    the image belonging to the negative class.

    Args:
        imgs_preprocessed: Batch of pre-processed images.
            Shape: [batch_size, channels, height, width]
        model: Model to use.
        requires_grad: Whether to calculate gradients. Defaults to True.

    Returns:
        probs: Tensor containing the probability of each possible label for each image in the batch.
            Shape: [batch_size, num_classes]
    """
    # Calculate logits and convert to probabilities.
    if requires_grad:
        logits = model(imgs_preprocessed)  # batch_size x num_classes
    else:
        with torch.no_grad():
            logits = model(imgs_preprocessed)

    if logits.shape[1] == 1:
        # We want to pretend like we have 2 output classes, so we need to apply a
        # sigmoid to the positive class and that set the negative class to 1 - sigmoid.
        return torch.cat((1 - F.sigmoid(logits), F.sigmoid(logits)), dim=1)

    return F.softmax(logits, dim=1)  # convert to probabilities
