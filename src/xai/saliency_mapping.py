from typing import cast

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.xai.xai_helpers import predict_labels


def saliency_mapping_helper(imgs_preprocessed: torch.Tensor, model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate gradients using Saliency Mapping.

    Args:
        imgs_preprocessed: Batch of pre-processed images.
            Shape: [batch_size, channels, height, width]
        model: Model to explain.

    Returns:
        gradients: Gradients of the model's output w.r.t. the input images.
            Shape: [batch_size, channels, height, width]
        labels_pred: Predicted labels.
            Shape: [batch_size]
    """
    # Set the requires_grad flag to True to calculate gradients.
    imgs_preprocessed = imgs_preprocessed.requires_grad_(True)

    # Predict labels.
    probs = predict_labels(imgs_preprocessed, model, True)

    # Get the indices of the target labels.
    labels_pred = torch.argmax(probs, dim=1)

    # Calculate gradients for each image in the batch.
    gradients = []
    for i in range(len(labels_pred)):
        model.zero_grad()
        probs[i, labels_pred[i]].backward(retain_graph=True)
        gradient = cast(torch.Tensor, imgs_preprocessed.grad)[i].detach()
        gradients.append(gradient)
    gradients = torch.stack(gradients)

    return gradients, labels_pred  # type: ignore


def saliency_mapping(imgs_preprocessed: torch.Tensor, model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate heatmaps using Saliency Mapping.

    Args:
        imgs_preprocessed: Batch of pre-processed images.
            Shape: [batch_size, channels, height, width]
        model: Model to explain.

    Returns:
        heatmap: Heatmap of the model's output w.r.t. the input images.
            Shape: [batch_size, height, width]
        labels_pred: Predicted labels.
            Shape: [batch_size]
    """
    # Calculate gradients for each image in the batch.
    gradients = []
    labels_pred = []

    for i in tqdm(range(len(imgs_preprocessed)), unit="image"):
        gradient, label_pred = saliency_mapping_helper(imgs_preprocessed[i].unsqueeze(0), model)
        gradients.append(gradient)
        labels_pred.append(label_pred)

    gradients = torch.cat(gradients)
    labels_pred = torch.cat(labels_pred)

    # Calculate heatmaps.
    heatmap = torch.abs(gradients).max(dim=1).values

    return heatmap, labels_pred
