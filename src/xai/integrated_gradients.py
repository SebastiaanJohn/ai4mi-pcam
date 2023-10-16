from typing import cast

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.xai.xai_helpers import predict_labels


def integrated_gradients_helper(
    imgs_preprocessed: torch.Tensor, model: nn.Module, baselines_preprocessed: torch.Tensor, steps: int = 20
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate heatmaps using Integrated Gradients.

    Args:
        imgs_preprocessed: Batch of pre-processed images.
            Shape: [batch_size, channels, height, width]
        model: Model to explain.
        baselines_preprocessed: Batch of pre-processed baseline images.
            Shape: [batch_size, channels, height, width]
        steps: Number of steps between the baseline and the input. Defaults to 20.

    Returns:
        heatmap: Heatmap of the model's output w.r.t. the input images.
            Shape: [batch_size, height, width, channels]
        labels_pred: Predicted labels.
            Shape: [batch_size]
    """
    # Predict labels.
    probs = predict_labels(imgs_preprocessed, model, True)

    # Get the indices of the target labels.
    labels_pred = torch.argmax(probs, dim=1)

    # Calculate gradients for each image in the batch.
    gradients = torch.zeros_like(imgs_preprocessed)
    diffs_preprocessed = (imgs_preprocessed.detach() - baselines_preprocessed) / steps
    partial_imgs_preprocessed = baselines_preprocessed.clone()
    for _ in tqdm(range(steps), unit="step"):
        partial_imgs_preprocessed += diffs_preprocessed
        partial_imgs_preprocessed_cp = (
            partial_imgs_preprocessed.clone().detach().requires_grad_(True)
        )  # Make tensor a leaf node.
        partial_probs = predict_labels(partial_imgs_preprocessed_cp, model, True)
        for i in range(len(labels_pred)):
            model.zero_grad()
            partial_probs[i, labels_pred[i]].backward(retain_graph=True)
            gradient = cast(torch.Tensor, partial_imgs_preprocessed_cp.grad)[i].detach()
            gradients[i] += gradient

    # Average the gradients across the steps and multiply by the diffs.
    # This line is sort of a "hack": it performs the averaging operation and
    # the multiplication by (x - x') in one step. It works because the
    # gradients were already summed across the steps, and the diffs were
    # already divided by `steps` when they were initialized.
    gradients *= diffs_preprocessed

    # Permute the dimensions to match the input images.
    gradients = gradients.permute(0, 2, 3, 1)  # b x c x h x w -> b x h x w x c

    return gradients, labels_pred


def integrated_gradients(imgs_preprocessed: torch.Tensor, model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate heatmaps using Integrated Gradients.

    Args:
        imgs_preprocessed: Batch of pre-processed images.
            Shape: [batch_size, channels, height, width]
        model: Model to explain.

    Returns:
        heatmap: Heatmap of the model's output w.r.t. the input images.
            Shape: [batch_size, height, width, channels]
        labels_pred: Predicted labels.
            Shape: [batch_size]
    """
    baselines_preprocessed = torch.zeros_like(imgs_preprocessed)
    return integrated_gradients_helper(imgs_preprocessed, model, baselines_preprocessed)
