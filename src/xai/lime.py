import skimage.segmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm

from src.xai.xai_helpers import predict_labels


def generate_segmentation_map(
    img: torch.Tensor,
    device: torch.device,
    segmap_ratio: float = 1.0,
    segmap_kernel_size: int = 5,
    segmap_max_dist: int = 0,
) -> torch.Tensor:
    """Generate a segmentation map for the given image.

    The segmentation map is a map of the same size as the image, where each
    pixel is assigned to a superpixel index. The superpixels are generated
    using the quickshift algorithm.

    Args:
        img: Image to generate the segmentation map for.
            Shape: [channels, height, width]
        device: Device to run the model on.
        segmap_ratio: Ratio between color-space proximity and image-space proximity.
            Must be between 0 and 1. Higher values give more weight to color-space. Defaults to 1.0.
        segmap_kernel_size: Width of Gaussian kernel used in smoothing the sample density.
            Higher means fewer clusters. Defaults to 5.
        segmap_max_dist: Cut-off point for data distances. Higher means fewer clusters.
            If 0, selects min(img_width, img_height) as the cut-off point. Defaults to 0.

    Returns:
        segmap: Integer segmentation map indicating which to superpixel each pixel belongs.
            Shape: [height, width]
    """
    segmap_max_dist = segmap_max_dist or min(img.shape[1:])
    return torch.from_numpy(
        skimage.segmentation.quickshift(
            img, channel_axis=0, ratio=segmap_ratio, kernel_size=segmap_kernel_size, max_dist=segmap_max_dist
        )
    ).to(device)


def sample_perturbations(num_perturbations: int, num_superpixels: int, device: torch.device) -> torch.Tensor:
    """Sample a number of perturbations.

    Note: One of the perturbations is always the unperturbed image.
    A 1 signifies the superpixel is on, a 0 signifies it is off.

    Args:
        num_perturbations: Number of perturbations to sample.
        num_superpixels: Number of superpixels in the image.
        device: Device to run the model on.

    Returns:
        perturbations: All perturbation vectors.
            Shape: [num_perturbations, num_superpixels]
    """
    perturbations = torch.bernoulli(torch.full((num_perturbations, num_superpixels), 0.5)).to(device)
    perturbations[0] = 1.0  # unperturbed image
    return perturbations


def compute_distances(perturbations: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Compute cosine distances between the perturbations and original image.

    Args:
        perturbations: All perturbation vectors.
            Shape: [num_perturbations, num_superpixels]
        device: Device to run the model on.

    Returns:
        distances: Cosine distances.
            Shape: [num_perturbations]
    """
    return F.cosine_similarity(perturbations, torch.tensor(1, device=device), dim=1)


def compute_weights(perturbations: torch.Tensor, device: torch.device, kernel_sigma: float = 0.25) -> torch.Tensor:
    """Compute the weights for the perturbations.

    The weights are calculated as follows:
        w_i = e^(-d_i^2 / kernel_sigma^2)
    where d_i is the cosine distance between the i-th perturbation and the original image.

    Args:
        perturbations: All perturbation vectors.
            Shape: [num_perturbations, num_superpixels]
        device: Device to run the model on.
        kernel_sigma: Sigma parameter for the kernel function.
            Defaults to 0.25.

    Returns:
        weights: Weights for the perturbations.
            Shape: [num_perturbations]
    """
    distances = compute_distances(perturbations, device)
    return torch.exp(-(distances**2) / kernel_sigma**2)


def perturb_img(
    img: torch.Tensor, segmap: torch.Tensor, perturbations: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Perturb the image by blacking out all superpixels marked as 0.

    Args:
        img: Image to perturb.
            Shape: [channels, height, width]
        segmap: Segmentation map.
            Shape: [height, width]
        perturbations: Batch of perturbation vectors.
            Shape: [batch_size, num_superpixels]
        device: Device to run the model on.

    Returns:
        imgs_perturbed: Perturbed images as tensor.
            Shape: [batch_size, channels, height, width]
    """
    superpixel_nums = torch.arange(perturbations.shape[1]).to(device)  # s
    superpixel_masks = (segmap == superpixel_nums.reshape(-1, 1, 1)).to(torch.float32)  # s x h x w
    masks = torch.einsum("bs,shw->bhw", perturbations, superpixel_masks)  # b x h x w
    return masks.unsqueeze(1) * img  # b x c x h x w


def generate_dataset(
    img: torch.Tensor,
    segmap: torch.Tensor,
    perturbations: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """Get the predicted probabilities for each perturbed image.

    Args:
        img: Image tensor.
            Shape: [channels, height, width]
        segmap: Segmentation map tensor.
            Shape: [height, width]
        perturbations: All perturbation vectors.
            Shape: [num_perturbations, num_superpixels]
        model: Model to use for prediction.
        device: Device to run the model on.
        batch_size: Number of perturbed images to generate at a time.
            Keep this low when running on CPU. Defaults to 16.

    Returns:
        probs_pred: Predicted probabilities for each class for each perturbed image.
            Shape: [num_perturbations, num_classes]
    """
    probs_pred = []
    for i in tqdm(range(0, perturbations.shape[0], batch_size), unit="batch", desc="Generating perturbations"):
        perturbations_batch = perturbations[i : i + batch_size]
        imgs_perturbed = perturb_img(img, segmap, perturbations_batch, device)
        probs_batch = predict_labels(imgs_perturbed, model, False)
        probs_pred.append(probs_batch)
    return torch.concatenate(probs_pred)


def train_linear_model(
    perturbations: torch.Tensor,
    weights: torch.Tensor,
    probs_pred: torch.Tensor,
    label_target: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Train a linear model to predict the top-5 labels from the perturbations.

    Args:
        perturbations: All perturbation vectors.
            Shape: [num_perturbations, num_superpixels]
        weights: Weights for the perturbations.
            Shape: [num_perturbations]
        probs_pred: Predicted probabilities for each class for each perturbed image.
            Shape: [num_perturbations, num_classes]
        label_target: Single target label to train the linear model for.
            Shape: []
        device: Device to run the model on.

    Returns:
        coefficients: Coefficients of the linear model.
            Shape: [num_superpixels]
    """
    model = LinearRegression().fit(perturbations, probs_pred[:, label_target], weights)
    return torch.from_numpy(model.coef_).to(device)


def create_heatmap(segmap: torch.Tensor, coefficients: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Create a heatmap from the coefficients of the linear model.

    Args:
        segmap: Segmentation map tensor.
            Shape: [height, width]
        coefficients: Coefficients of the linear model.
            Shape: [num_superpixels]
        device: Device to run the model on.

    Returns:
        heatmap: Heatmap of the model's output w.r.t. the input image.
            Shape: [height, width]
    """
    superpixel_nums = torch.arange(coefficients.shape[0]).to(device)  # s
    superpixel_masks = (segmap == superpixel_nums.reshape(-1, 1, 1)).to(torch.float32)  # s x h x w
    return torch.einsum("s,shw->hw", coefficients, superpixel_masks)  # h x w


def lime_helper(
    imgs_preprocessed: torch.Tensor,
    model: nn.Module,
    num_perturbations: int = 300,
    batch_size: int = 16,
    segmap_ratio: float = 1.0,
    segmap_kernel_size: int = 5,
    segmap_max_dist: int = 0,
    kernel_sigma: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pipeline for the LIME algorithm.

    Args:
        imgs_preprocessed: Batch of pre-processed images.
            Shape: [batch_size, channels, height, width]
        model: Model to explain.
        num_perturbations: Number of perturbations to sample.
            Defaults to 300.
        batch_size: Number of perturbed images to generate at a time.
            Keep this low when running on CPU. Defaults to 16.
        segmap_ratio: Ratio between color-space proximity and image-space proximity.
            Must be between 0 and 1. Higher values give more weight to color-space. Defaults to 1.0.
        segmap_kernel_size: Width of Gaussian kernel used in smoothing the sample density.
            Higher means fewer clusters. Defaults to 5.
        segmap_max_dist: Cut-off point for data distances. Higher means fewer clusters.
            If 0, selects min(img_width, img_height) as the cut-off point. Defaults to 0.
        kernel_sigma: Sigma parameter for the kernel function. Defaults to 0.25.

    Returns:
        heatmap: Heatmap of the model's output w.r.t. the input images.
            Shape: [batch_size, height, width]
        labels_pred: Predicted labels.
            Shape: [batch_size]
    """
    device = imgs_preprocessed.device

    # Generate segmentation maps.
    all_segmap = [
        generate_segmentation_map(img, device, segmap_ratio, segmap_kernel_size, segmap_max_dist)
        for img in imgs_preprocessed
    ]
    all_num_superpixels = [segmap.unique(sorted=False).shape[0] for segmap in all_segmap]

    # Sample perturbations.
    all_perturbations = [
        sample_perturbations(num_perturbations, num_superpixels, device) for num_superpixels in all_num_superpixels
    ]

    # Get weights for each perturbation.
    all_weights = [compute_weights(perturbations, device, kernel_sigma) for perturbations in all_perturbations]

    # Predict class probabilities for all perturbations of each image.
    all_probs_pred = [
        generate_dataset(img, segmap, perturbations, model, device, batch_size)
        for img, segmap, perturbations in zip(imgs_preprocessed, all_segmap, all_perturbations)
    ]

    # We only need the labels for the unperturbed images, which are always at index 0.
    labels_pred = torch.stack([probs_pred[0] for probs_pred in all_probs_pred]).argmax(dim=1)

    # Train the linear models.
    all_coefficients = [
        train_linear_model(perturbations, weights, probs_pred, label_pred, device)
        for perturbations, probs_pred, weights, label_pred in zip(
            all_perturbations, all_probs_pred, all_weights, labels_pred
        )
    ]

    # Create the heatmaps.
    heatmap = torch.stack(
        [create_heatmap(segmap, coefficients, device) for segmap, coefficients in zip(all_segmap, all_coefficients)]
    )

    return heatmap, labels_pred


def lime(imgs_preprocessed: torch.Tensor, model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate heatmaps using LIME.

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
    

    return lime_helper(imgs_preprocessed, model)
