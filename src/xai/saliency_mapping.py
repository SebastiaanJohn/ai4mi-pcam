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
    # Calculate gradients for each image in the batch.
    gradients = []
    labels_pred = []

    for i in tqdm(range(len(imgs_preprocessed)), unit="image"):
        img_preprocessed = imgs_preprocessed[i].unsqueeze(0)

        # Set the requires_grad flag to True to calculate gradients.
        img_preprocessed = img_preprocessed.requires_grad_(True)

        # Predict labels.
        model.zero_grad()
        probs = predict_labels(img_preprocessed, model, True)

        # Get the indices of the target labels.
        label_pred = torch.argmax(probs, dim=1)[0]

        # Calculate gradients.
        probs[0, label_pred].backward(retain_graph=True)
        gradient = cast(torch.Tensor, img_preprocessed.grad)[0].detach()

        labels_pred.append(label_pred)
        gradients.append(gradient)

    gradients = torch.stack(gradients)
    labels_pred = torch.stack(labels_pred)

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
    # Calculate gradient maps.
    gradients, labels_pred = saliency_mapping_helper(imgs_preprocessed, model)

    # Calculate heatmaps.
    heatmap = torch.abs(gradients).max(dim=1).values

    return heatmap, labels_pred
