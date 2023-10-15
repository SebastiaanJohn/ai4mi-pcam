import argparse
import logging
from collections.abc import Callable, Sequence
from math import ceil
from pathlib import Path
from typing import Generic, TypeAlias, TypeVar, cast

import matplotlib
import matplotlib.axes
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from lime import lime_image
from torchvision.models import densenet121, resnet34, resnet50
from torchvision.transforms import transforms
from tqdm.auto import tqdm

from src.config import settings
from src.datasets.PCAM.datamodule import PCAMDataModule


ExplanationMethod: TypeAlias = Callable[[torch.Tensor, nn.Module], tuple[torch.Tensor, torch.Tensor]]


DType = TypeVar("DType")


class NDArrayGeneric(np.ndarray, Generic[DType]):
    """np.ndarray that allows for static type hinting of generics."""

    def __getitem__(self, key) -> DType:
        return super().__getitem__(key)  # type: ignore


def select_explanation_method(explanation_method_name: str) -> ExplanationMethod:
    """Get the function associated with the given explanation method name.

    Args:
        explanation_method_name: The explanation method name.

    Returns:
        explanation_method: The explanation method's function.
    """
    if explanation_method_name == "lime":
        explanation_method = lime
    elif explanation_method_name == "integrated_gradients":
        explanation_method = integrated_gradients
    elif explanation_method_name == "saliency_mapping":
        explanation_method = saliency_mapping
    else:
        raise ValueError(f"Unknown explanation method: {explanation_method_name}")
    return explanation_method


def load_model(model_name: str) -> nn.Module:
    """Load the model associated with the given model name.

    Args:
        model_name: Model name.

    Returns:
        model: PyTorch model.
    """
    if model_name == "resnet34":
        model = resnet34()
    elif model_name == "resnet50":
        model = resnet50()
    elif model_name == "densenet121":
        model = densenet121()
    elif model_name == "vf-cnn":
        model = vfcnn()  # TODO
    else:
        raise ValueError(f"Unknown model: {model_name}")

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    model.eval()

    # TODO load model from checkpoint
    # MODEL_PATH = "./src/models/checkpoints/best-loss-model-epoch=00-val_loss=0.83.ckpt"

    # _ = PCAMSystem.load_from_checkpoint(checkpoint_path=MODEL_PATH, model=model)

    return model


def init_dataloader(batch_size: int) -> data.DataLoader:
    """Initialize the dataloader for the test set.

    Args:
        batch_size: The batch size to use.

    Returns:
        test_dataloader: Dataloader for the PCAM test set.
    """
    datamodule = PCAMDataModule(data_dir=settings.raw_data_dir, batch_size=batch_size, lazy_loading=True)
    datamodule.setup(stage="predict")
    test_dataloader = datamodule.test_dataloader()
    return test_dataloader


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
        # We want to pretend like we have 2 output classes, so we need to
        # apply a sigmoid and after that set the second class to 1 - sigmoid.
        return torch.cat((F.sigmoid(logits), 1 - F.sigmoid(logits)), dim=1)
    else:
        return F.softmax(logits, dim=1)  # convert to probabilities


def lime(imgs_preprocessed: torch.Tensor, model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate the heatmap using LIME.

    Args:
        imgs_preprocessed: Batch of pre-processed images.
            Shape: [batch_size, channels, height, width]
        model: Model to use.

    Returns:
        heatmap: Heatmap of the model's output w.r.t. the input images.
            Shape: [batch_size, height, width, channels]
        labels_pred: Predicted labels.
            Shape: [batch_size]
    """
    pill_transf = transforms.Compose(
        [
            transforms.ToPILImage(),
            # transforms.Resize((256, 256)),
            # transforms.CenterCrop(224)
        ]
    )

    preprocess = transforms.Compose([transforms.ToTensor()])

    def batch_predict(batch):
        batch = torch.stack(tuple(preprocess(i) for i in batch), dim=0)
        logits = model(batch)
        probs = torch.cat((F.sigmoid(logits), 1 - F.sigmoid(logits)), dim=1)
        return probs.detach().cpu().numpy()

    explainer = lime_image.LimeImageExplainer()

    labels_pred = torch.tensor([])
    explanation_masks = torch.tensor([])
    for image in imgs_preprocessed:
        explanation = explainer.explain_instance(
            np.array(pill_transf(image)), batch_predict, top_labels=2, num_samples=1000
        )

        _, mask = explanation.get_image_and_mask(1, positive_only=True, num_features=2)
        pred_label = torch.tensor([explanation.top_labels[0]])

        labels_pred = torch.cat((labels_pred, pred_label), dim=0)
        explanation_masks = torch.cat((explanation_masks, torch.tensor([mask])), dim=0)

    heatmap = explanation_masks.unsqueeze(-1)

    return heatmap, labels_pred


def saliency_mapping(imgs_preprocessed: torch.Tensor, model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate the gradients using Saliency Mapping.

    Args:
        imgs_preprocessed: Batch of pre-processed images.
            Shape: [batch_size, channels, height, width]
        model: Model to use.

    Returns:
        heatmap: Heatmap of the model's output w.r.t. the input images.
            Shape: [batch_size, height, width, channels]
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
    for i in tqdm(range(len(labels_pred)), unit="image"):
        model.zero_grad()
        probs[i, labels_pred[i]].backward(retain_graph=True)
        gradient = cast(torch.Tensor, imgs_preprocessed.grad)[i].detach()
        gradients.append(gradient)
    gradients = torch.stack(gradients)
    gradients = gradients.permute(0, 2, 3, 1)  # b x c x h x w -> b x h x w x c

    return gradients, labels_pred


def integrated_gradients_helper(
    imgs_preprocessed: torch.Tensor, model: nn.Module, baselines_preprocessed: torch.Tensor, steps: int = 20
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate the gradients using Integrated Gradients.

    Args:
        imgs_preprocessed: Batch of pre-processed images.
            Shape: [batch_size, channels, height, width]
        model: Model to use.
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
    """Calculate the gradients using IG with a black image baseline.

    Args:
        imgs_preprocessed: Batch of pre-processed images.
            Shape: [batch_size, channels, height, width]
        model: Model to use.

    Returns:
        heatmap: Heatmap of the model's output w.r.t. the input images.
            Shape: [batch_size, height, width, channels]
        labels_pred: Predicted labels.
            Shape: [batch_size]
    """
    baselines_preprocessed = torch.zeros_like(imgs_preprocessed)
    return integrated_gradients_helper(imgs_preprocessed, model, baselines_preprocessed)


def explain_model(
    explanation_method_name: str,
    model_name: str,
    test_dataloader: data.DataLoader,
    device: torch.device,
    batch_size: int,
    num_images: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate and cache the predictions and heatmaps for a single model.

    Args:
        explanation_method_name: The name of the explanation method.
        model_names: List of model names to explain.
        test_dataloader: The dataloader for the test set.
        device: The device to use.
        batch_size: The batch size to use.
        num_images: The number of images to generate an explanation for.

    Returns:
        Tuple containing:
            labels_pred: Predicted labels.
                Shape: [num_images]
            heatmaps: Heatmaps signifying what the model looks at.
                Shape: [num_images, height, width]
    """
    # Select the explanation method.
    explanation_method = select_explanation_method(explanation_method_name)

    # Load the model.
    model = load_model(model_name)
    model = model.to(device)

    # Determine where the results should be cached.
    cache_path = Path(f"./data/results/{explanation_method_name}_{model_name}.pt")

    if cache_path.exists() and False:  # TODO remove False
        # If the results for this model have already been calculated, retrieve the cache.
        logging.info(f"Loading cache for {model_name}...")
        labels_pred, heatmaps = torch.load(cache_path)
        start_at = len(labels_pred)
        logging.info(f"Number of results loaded: {start_at}")
        if start_at >= num_images:
            # If the cache contains enough results, we're done.
            logging.info("Cache contains enough results. Skipping.")
            return torch.concatenate(labels_pred), torch.concatenate(heatmaps)
    else:
        logging.info("No cache found.")
        labels_pred = []
        heatmaps = []
        start_at = 0

    # Now calculate the rest of the results.
    logging.info(f"Calculating results for {model_name}...")

    idx = 0
    for imgs_preprocessed_batch, _ in tqdm(test_dataloader, unit="batch", total=ceil(num_images / batch_size)):
        # Check if we need to skip (a part of) this batch.
        batch_size = len(imgs_preprocessed_batch)
        if idx == num_images:
            # We're done.
            break
        if idx + batch_size <= start_at:
            # Skip the whole batch.
            idx += batch_size
            continue
        elif idx < start_at < idx + batch_size:
            # Skip the first few images in the batch.
            imgs_preprocessed_batch = imgs_preprocessed_batch[start_at - idx :]
            idx = start_at
            batch_size = len(imgs_preprocessed_batch)
        if num_images < idx + batch_size:
            # Skip the last few images in the batch.
            imgs_preprocessed_batch = imgs_preprocessed_batch[: num_images - idx]
            batch_size = len(imgs_preprocessed_batch)

        imgs_preprocessed_batch = imgs_preprocessed_batch.to(device)

        # Calculate the heatmaps.
        heatmaps_batch, labels_pred_batch = explanation_method(imgs_preprocessed_batch, model)

        # Average across channels to get a single value per pixel.
        heatmaps_batch = torch.mean(heatmaps_batch, dim=-1)  # b x h x w x c -> b x h x w

        # Normalize the heatmaps.
        heatmaps_agg = heatmaps_batch.reshape(batch_size, -1)  # pytorch doesn't support min/max over multiple dims
        heatmaps_min = torch.min(heatmaps_agg, dim=-1)[0].reshape(-1, 1, 1)
        heatmaps_max = torch.max(heatmaps_agg, dim=-1)[0].reshape(-1, 1, 1)
        heatmaps_batch = (heatmaps_batch - heatmaps_min) / (heatmaps_max - heatmaps_min)  # normalize to [0, 1]
        heatmaps_batch /= heatmaps_batch.sum()  # normalize to sum to 1

        # Append the batches.
        labels_pred.append(labels_pred_batch)
        heatmaps.append(heatmaps_batch)

        # Save the results.
        torch.save((labels_pred, heatmaps), cache_path)

        # Update the index.
        idx += batch_size

    return torch.concatenate(labels_pred), torch.concatenate(heatmaps)


def explain_models(
    explanation_method_name: str,
    model_names: list[str],
    test_dataloader: data.DataLoader,
    device: torch.device,
    batch_size: int,
    num_images: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Calculate and cache the predictions and heatmaps for each model.

    Args:
        explanation_method_name: The name of the explanation method.
        model_names: List of model names to explain.
        test_dataloader: The dataloader for the test set.
        device: The device to use.
        batch_size: The batch size to use.
        num_images: The number of images to generate an explanation for.

    Returns:
        Tuple containing:
            labels_pred: Predicted labels.
                Length: num_models
                Shape of inner Tensors: [num_images]
            heatmaps: Heatmaps signifying what the model looks at.
                Length: num_models
                Shape of inner Tensors: [num_images, height, width]
    """
    # Calculate and cache the heatmaps for each model.
    labels_pred_models = []
    heatmaps_models = []

    for model_name in model_names:
        labels_pred, heatmaps = explain_model(
            explanation_method_name, model_name, test_dataloader, device, batch_size, num_images
        )

        # Append the results.
        labels_pred_models.append(labels_pred)
        heatmaps_models.append(heatmaps)

    return labels_pred_models, heatmaps_models


def weighted_iou(heatmaps1: torch.Tensor, heatmaps2: torch.Tensor) -> torch.Tensor:
    """Calculate the weighted IoU between two sets of heatmaps.

    Args:
        heatmaps1: The first set of heatmaps.
            Shape: [*, height, width]
        heatmaps2: The second set of heatmaps.
            Shape: [*, height, width]

    Returns:
        iou: The weighted IoU for each pair of heatmaps.
            Shape: [*]
    """
    # Calculate the intersection and union for each pair of heatmaps.
    intersections = torch.sum(torch.min(heatmaps1, heatmaps2), dim=(-2, -1))
    unions = torch.sum(torch.max(heatmaps1, heatmaps2), dim=(-2, -1))

    # Calculate the weighted IoU.
    return intersections / unions


def calculate_agreement(heatmaps: list[torch.Tensor]) -> np.ndarray:
    """Calculate the agreement between each pair of models.

    Note: The "agreement" is defined as the average weighted IoU between the heatmaps.

    Args:
        heatmaps: List of heatmaps.
            Length: num_models
            Shape of Tensors: [num_images, height, width]
        model_names: List of model names.

    Returns:
        agreement_table: The similarity table.
            Shape: [num_models, num_models]
    """
    # Calculate the overall agreement between each pair of models.
    agreement_table = np.zeros((len(heatmaps), len(heatmaps)))
    for i in range(len(heatmaps)):
        agreement_table[i, i] = 1.0
        for j in range(i + 1, len(heatmaps)):
            agreement_table[i, j] = torch.mean(weighted_iou(heatmaps[i], heatmaps[j])).item()
            agreement_table[j, i] = agreement_table[i, j]

    return agreement_table


def get_scalar_mappable(
    values: Sequence[float],
    from_color: str = "red",
    to_color: str = "green",
    use_log_scale: bool = False,
    zero_is_white: bool = False,
) -> matplotlib.cm.ScalarMappable:
    """Get a ScalarMappable with color map: from_color -> white -> to_color.

    Args:
        values: The values to map to colors.
        from_color: The color to map the smallest value to.
        to_color: The color to map the largest value to.
        use_log_scale: Whether to use a log scale.
        zero_is_white: Whether to map zero values to white.

    Returns:
        scalar_mappable: ScalarMappable color map for the values.
    """
    values_arr = np.array(values)
    vmin = values_arr.min()
    vmax = values_arr.max()

    # If zero values should be mapped to white, then we need to make sure that
    # there is at least one negative value and one positive value.
    if zero_is_white:
        if vmin > 0:
            vmin = -vmin
        elif vmax < 0:
            vmax = -vmax

    # Use a log scale if specified, otherwise use a linear scale.
    if use_log_scale:
        norm = matplotlib.colors.AsinhNorm(
            vmin=vmin,
            vmax=vmax,
            # Choose the value that is closest to zero, but not zero itself.
            linear_width=(vmin if vmin > 0 else np.abs(values_arr[values_arr != 0]).min()),  # type: ignore
        )
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # Map zero values to white if specified, otherwise map them to the middle of the color map.
    white_value = norm(0) if zero_is_white else 0.5

    # Create a green-white-red color map, or invert it if specified.
    color_list = [(0, from_color), (white_value, "white"), (1, to_color)]

    # Create the ScalarMappable color map.
    return matplotlib.cm.ScalarMappable(
        norm=norm, cmap=(matplotlib.colors.LinearSegmentedColormap.from_list("x_white_y", color_list))
    )


def display_matrix(
    matrix: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    xticks: np.ndarray,
    yticks: np.ndarray,
    xticklabels: list[str],
    yticklabels: list[str],
) -> None:
    """Show a heatmap of the matrix.

    Args:
        matrix: The matrix to show.
            Shape: [num_rows, num_cols]
        title: Title of the plot.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        xticks: The positions of the ticks on the x-axis.
            Shape: [num_cols]
        yticks: The positions of the ticks on the y-axis.
            Shape: [num_rows]
        xticklabels: The labels for the ticks on the x-axis.
            Length: num_cols
        yticklabels: The labels for the ticks on the y-axis.
            Length: num_rows
    """
    # Get color map for both matrices.
    scalar_mappable = get_scalar_mappable([0, 1], from_color="red", to_color="green")

    # Plot the matrices. Use a red-white-green colormap to show how similar the models' explanations are.
    fig, ax = plt.subplots(
        1, 1, figsize=(3 + min(4, matrix.shape[1] * 0.5), 2 + min(4, matrix.shape[0] * 0.5)), squeeze=False
    )
    ax = cast(matplotlib.axes.Axes, ax)

    ax.imshow(matrix.T, cmap=scalar_mappable.get_cmap(), norm=scalar_mappable.norm)
    for x in range(matrix.shape[1]):
        for y in range(matrix.shape[0]):
            ax.text(x, y, round(matrix[y, x], 2), ha="center", va="center", color="black", fontsize=14)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels, rotation=90, va="center")

    fig.colorbar(scalar_mappable, ax=ax)
    plt.tight_layout()
    plt.show()


def visualize_agreement_table(agreement_table: np.ndarray, model_names: list[str]) -> None:
    """Visualize the agreement table.

    Args:
        agreement_table: The agreement table.
            Shape: [num_models, num_models]
        model_names: List of model names.
            Length: num_models
    """
    display_matrix(
        agreement_table,
        "Agreement between models\n(Mean Weighted IoU)",
        "Model 1",
        "Model 2",
        np.arange(len(model_names)),
        np.arange(len(model_names)),
        model_names,
        model_names,
    )


def plot_heatmaps(
    img: torch.Tensor,
    gt_label: int,
    heatmap1: torch.Tensor,
    heatmap2: torch.Tensor,
    model_name1: str,
    model_name2: str,
    pred_label1: int,
    pred_label2: int,
    title: str,
) -> None:
    """Plot the heatmap of the given image.

    Args:
        img: Image to plot the heatmap for.
            Shape: [channels, height, width]
        gt_label: Ground truth label.
        heatmap1: Heatmap for the first model.
            Shape: [height, width]
        heatmap2: Heatmap for the second model.
            Shape: [height, width]
        model_name1: Name of the first model.
        model_name2: Name of the second model.
        pred_label1: Predicted label for the first model.
        pred_label2: Predicted label for the second model.
        title: Title of the plot.
    """
    img = torch.permute(img, (1, 2, 0))  # c x h x w -> h x w x c

    fig, axs = plt.subplots(1, 5, figsize=(21, 4))
    axs = cast(NDArrayGeneric[matplotlib.axes.Axes], axs)
    fig.suptitle(f"{title}, IoU = {weighted_iou(heatmap1, heatmap2).item():.2f}", fontsize=20)

    vmax = max(torch.max(heatmap1).item(), torch.max(heatmap2).item())
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=vmax), cmap="hot")
    axs[0].imshow(img, cmap=sm.get_cmap(), norm=sm.norm)
    axs[0].axis("off")
    axs[0].set_title(f"Original image (gt label: {gt_label})")
    axs[1].imshow(heatmap1, cmap=sm.get_cmap(), norm=sm.norm)
    axs[1].axis("off")
    axs[1].set_title(f"Heatmap of {model_name1} (pred label: {pred_label1})")
    axs[2].imshow(heatmap2, cmap=sm.get_cmap(), norm=sm.norm)
    axs[2].axis("off")
    axs[2].set_title(f"Heatmap of {model_name2} (pred label: {pred_label2})")
    axs[3].imshow(torch.min(heatmap1, heatmap2), cmap=sm.get_cmap(), norm=sm.norm)
    axs[3].axis("off")
    axs[3].set_title("Intersection")
    axs[4].imshow(torch.max(heatmap1, heatmap2), cmap=sm.get_cmap(), norm=sm.norm)
    axs[4].axis("off")
    axs[4].set_title("Union")

    fig.colorbar(sm, ax=axs[-1])
    plt.tight_layout()
    plt.show()


def main(args: argparse.Namespace) -> None:
    """The entry point of the program.

    Args:
        args: The command line arguments.
    """
    if len(args.models) < 2:
        raise ValueError(f"Expected at least 2 models, got {len(args.models)}")

    if len(args.models) > 2 and args.plot_heatmaps:
        raise ValueError(f"Expected exactly 2 models when plotting heatmaps, got {len(args.models)}")

    # Select device.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Create the dataloader.
    test_dataloader = init_dataloader(args.batch_size)
    if args.num_images > len(test_dataloader):
        raise ValueError(f"Expected at most {len(test_dataloader)} images, got {args.num_images}")

    # Calculate and cache the heatmaps for each model.
    logging.info("Calculating heatmaps...")
    labels_pred, heatmaps = explain_models(
        args.explanation, args.models, test_dataloader, device, args.batch_size, args.num_images
    )

    if args.plot_heatmaps:
        for i in range(args.num_images):
            plot_heatmaps(
                test_dataloader.dataset[i][0],
                test_dataloader.dataset[i][1].item(),
                heatmaps[0][i],
                heatmaps[1][i],
                args.models[0],
                args.models[1],
                int(labels_pred[0][i].item()),
                int(labels_pred[1][i].item()),
                f"{args.explanation.replace('_', ' ').capitalize()}, Image #{i + 1}",
            )
    else:
        # Calculate the agreement between each pair of models.
        agreement_table = calculate_agreement(heatmaps)

        # Visualize the agreement table.
        visualize_agreement_table(agreement_table, args.models)


if __name__ == "__main__":
    # Create the argument parser.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="XAI Comparison"
    )

    # Define command line arguments.
    # fmt: off
    parser.add_argument(
        "--logging_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="The logging level to use.",
    )
    parser.add_argument(
        "--models",
        type=str,
        choices=["resnet34", "resnet50", "densenet121", "vf-cnn"],
        nargs="+",
        required=True,
        help="The models to compare (must be at least 2).",
    )
    parser.add_argument(
        "--explanation",
        type=str,
        choices=["lime", "integrated_gradients", "saliency_mapping"],
        required=True,
        help="The explanation method to use for generating heatmaps.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=100,
        help="The number of images to generate an explanation for."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="The batch size to use when generating explanations."
    )
    parser.add_argument(
        "--plot_heatmaps",
        action="store_true",
        help="Whether to plot the heatmaps. If True, you must specify 2 models."
    )
    # fmt: on

    args = parser.parse_args()

    # Configure the logger.
    logging.basicConfig(level=args.logging_level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    logging.debug(f"{args=}")

    main(args)
