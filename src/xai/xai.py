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
import torch.utils.data as data
from torchvision import transforms
from torchvision.models import densenet121, resnet34, resnet50, vit_b_16

from src.config import settings
from src.datasets.PCAM.datamodule import PCAMDataModule
from src.engines.system import PCAMSystem
from src.xai.integrated_gradients import integrated_gradients
from src.xai.lime import lime
from src.xai.saliency_mapping import saliency_mapping


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
    if explanation_method_name == "saliency_mapping":
        explanation_method = saliency_mapping
    elif explanation_method_name == "integrated_gradients":
        explanation_method = integrated_gradients
    elif explanation_method_name == "lime":
        explanation_method = lime
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
    elif model_name == "vit":
        model = vit_b_16()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name == "densenet121":
        num_features = cast(nn.Linear, model.classifier).in_features
        model.classifier = nn.Linear(num_features, 1)
    elif model_name == "vit":
        num_features = cast(nn.Linear, model.heads.head).in_features
        model.heads.head = nn.Linear(num_features, 1)
    else:
        num_features = cast(nn.Linear, model.fc).in_features
        model.fc = nn.Linear(num_features, 1)

    model.eval()

    MODEL_PATHS = {
        "resnet34": "./models/ResNet34/best-loss-model-epoch=00-val_loss=0.40.ckpt",
        "resnet50": "./models/ResNet50/best-loss-model-epoch=02-val_loss=0.43.ckpt",
        "densenet121": "./models/DenseNet121/best-loss-model-epoch=01-val_loss=0.37.ckpt",
        "vit": "./models/ViT/best-loss-model-epoch=01-val_loss=0.52.ckpt",
    }

    system = PCAMSystem.load_from_checkpoint(checkpoint_path=MODEL_PATHS[model_name], model=model, map_location="cpu")
    model = cast(nn.Module, system.model)

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
            heatmaps: Heatmaps signifying what the model looks at.
                Shape: [num_images, height, width]
            labels_pred: Predicted labels.
                Shape: [num_images]
    """
    # Select the explanation method.
    explanation_method = select_explanation_method(explanation_method_name)

    # Load the model.
    model = load_model(model_name)
    model = model.to(device)

    # Determine where the results should be cached.
    cache_path = Path(f"./data/results/{explanation_method_name}_{model_name}.pt")

    if cache_path.exists():
        # If the results for this model have already been calculated, retrieve the cache.
        logging.info(f"Loading cache for {model_name}...")
        heatmaps, labels_pred = torch.load(cache_path)
        start_at = sum(len(l) for l in labels_pred)  # type: ignore
        logging.info(f"Number of results loaded: {start_at}")
        if start_at >= num_images:
            # If the cache contains enough results, we're done.
            logging.info("Cache contains enough results. Skipping.")
            return torch.concatenate(heatmaps), torch.concatenate(labels_pred)
    else:
        logging.info(f"No cache found at {cache_path}.")
        heatmaps = []
        labels_pred = []
        start_at = 0

    # Now calculate the rest of the results.
    logging.info(f"Calculating heatmaps for {model_name}...")
    logging.info("NOTE: Intermediate results are automatically")
    logging.info("      saved to disk after every batch. You can")
    logging.info("      stop the calculation at any time, and the")
    logging.info("      program will resume from where it left off.")

    idx = 0
    for batch_idx, (imgs_preprocessed_batch, _) in enumerate(test_dataloader):
        # Check if we need to skip (a part of) this batch.
        curr_batch_size = len(imgs_preprocessed_batch)
        if idx == num_images:
            # We're done.
            break
        if idx + curr_batch_size <= start_at:
            # Skip the whole batch.
            idx += curr_batch_size
            continue
        elif idx < start_at < idx + curr_batch_size:
            # Skip the first few images in the batch.
            imgs_preprocessed_batch = imgs_preprocessed_batch[start_at - idx :]
            idx = start_at
            curr_batch_size = len(imgs_preprocessed_batch)
        if num_images < idx + curr_batch_size:
            # Skip the last few images in the batch.
            imgs_preprocessed_batch = imgs_preprocessed_batch[: num_images - idx]
            curr_batch_size = len(imgs_preprocessed_batch)

        logging.info(f"[ Current batch: {batch_idx + 1}/{ceil(num_images / batch_size)} ]")
        imgs_preprocessed_batch = imgs_preprocessed_batch.to(device)

        # Calculate the heatmaps.
        heatmaps_batch, labels_pred_batch = explanation_method(imgs_preprocessed_batch, model)

        # Normalize the heatmaps.
        heatmaps_agg = heatmaps_batch.reshape(curr_batch_size, -1)  # torch doesn't support min/max over multiple dims
        heatmaps_min = torch.min(heatmaps_agg, dim=-1).values.reshape(-1, 1, 1)
        heatmaps_max = torch.max(heatmaps_agg, dim=-1).values.reshape(-1, 1, 1)
        heatmaps_batch = (heatmaps_batch - heatmaps_min) / (heatmaps_max - heatmaps_min)  # normalize to [0, 1]
        heatmaps_batch /= heatmaps_batch.sum(dim=(-2, -1), keepdims=True)  # normalize to sum to 1

        # Append the batches.
        heatmaps.append(heatmaps_batch)
        labels_pred.append(labels_pred_batch)

        # Save the results.
        torch.save((heatmaps, labels_pred), cache_path)

        # Update the index.
        idx += curr_batch_size

    return torch.concatenate(heatmaps), torch.concatenate(labels_pred)


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
            heatmaps: Heatmaps signifying what the model looks at.
                Length: num_models
                Shape of inner Tensors: [num_images, height, width]
            labels_pred: Predicted labels.
                Length: num_models
                Shape of inner Tensors: [num_images]
    """
    # Calculate and cache the heatmaps for each model.
    heatmaps_models = []
    labels_pred_models = []

    for model_name in model_names:
        # Pretty print so that the output is always 80 characters wide.
        chars_left = 80 - 45 - len(model_name)
        logging.info("/" + " " * 64 + "\\")
        logging.info(
            f"| {'=' * (chars_left // 2)} "
            f"Calculating heatmaps for {model_name}"
            f" {'='* (chars_left - chars_left // 2)} |"
        )
        logging.info("\\" + " " * 64 + "/")

        if model_name == "vit":
            test_dataloader.dataset._transform = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            )
        else:
            test_dataloader.dataset._transform = None

        heatmaps, labels_pred = explain_model(
            explanation_method_name, model_name, test_dataloader, device, batch_size, num_images
        )

        if model_name == "vit":
            # Cast the heatmaps back to [b, 96, 96].
            print(f"{heatmaps.shape=}")
            transform = transforms.Compose(
                [transforms.ToPILImage(), transforms.Resize((96, 96)), transforms.ToTensor()]
            )
            heatmaps = torch.concatenate([transform(h) for h in heatmaps])
            print(f"{heatmaps.shape=}")

        # Append the results.
        heatmaps_models.append(heatmaps)
        labels_pred_models.append(labels_pred)

    return heatmaps_models, labels_pred_models


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
    fig, ax = plt.subplots(1, 1, figsize=(3 + min(4, matrix.shape[1] * 0.5), 2 + min(4, matrix.shape[0] * 0.5)))
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


def visualize_agreement_table(
    agreement_table: np.ndarray, explanation_method_name: str, model_names: list[str]
) -> None:
    """Visualize the agreement table.

    Args:
        agreement_table: The agreement table.
            Shape: [num_models, num_models]
        explanation_method_name: The name of the explanation method.
        model_names: List of model names.
            Length: num_models
    """
    title = f"{explanation_method_name.replace('_', ' ').capitalize()}"  # , Image #{i + 1}"
    display_matrix(
        agreement_table,
        f"{title}\nAgreement between models\n(Mean Weighted IoU)",
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
    explanation_method_name: str,
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
        explanation_method_name: The name of the explanation method.
    """
    img = torch.permute(img, (1, 2, 0))  # c x h x w -> h x w x c

    fig, axs = plt.subplots(1, 5, figsize=(21, 4))
    axs = cast(NDArrayGeneric[matplotlib.axes.Axes], axs)
    title = f"{explanation_method_name.replace('_', ' ').capitalize()}"  # , Image #{i + 1}"
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
    heatmaps, labels_pred = explain_models(
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
                args.explanation,
            )
    else:
        # Calculate the agreement between each pair of models.
        agreement_table = calculate_agreement(heatmaps)

        # Visualize the agreement table.
        visualize_agreement_table(agreement_table, args.explanation, args.models)


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
        choices=["resnet34", "resnet50", "densenet121", "vit"],
        nargs="+",
        required=True,
        help="The models to compare (must be at least 2).",
    )
    parser.add_argument(
        "--explanation",
        type=str,
        choices=["saliency_mapping", "integrated_gradients", "lime"],
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
