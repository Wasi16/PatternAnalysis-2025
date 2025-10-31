"""
utils.py 
--------

General utility functions for preprocessing and analysis.

Includes: 
- mean_std_calc(): Calculate the mean and standard deviation of ADNI image dataset.

"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 

def mean_std_calc(data_dir, batch_size=32, num_workers=2, grayscale=True, save_to_json=True, resize=(256, 256)):
    """
    Calculates the mean and standard deviation of an image dataset.
    These values are used for normalization during preprocessing.

    Args:
        data_dir (str): Path to dataset root directory (with class subfolders).
        batch_size (int): Number of images to process per batch.
        num_workers (int): Number of worker threads for data loading.
        grayscale (bool): Whether to convert images to grayscale.
        save_to_json (bool): Placeholder for saving values (not used in this version).
        resize (tuple): Target size (H, W) for resizing images.

    Returns:
        (tuple, tuple): Mean and standard deviation of image channels.
                        Example: (mean,), (std,) for grayscale.
    """

    transform_list = []

    if grayscale: 
        transform_list.append(transforms.Grayscale(num_output_channels=1))

    transform_list += [
        transforms.Resize(resize),
        transforms.ToTensor()
    ]

    transform = transforms.Compose(transform_list)
    # Load dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    mean = 0.0
    std = 0.0
    total_images = 0

    # Loop through dataset batches and compute mean and std for each batch
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)  # flatten [B, C, H, W] → [B, C, H*W]
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_images += batch_samples
    
    # Average values across all batches
    mean /= total_images
    std /= total_images

    mean_tuple = tuple(mean.tolist())
    std_tuple = tuple(std.tolist())

    print(f"Mean: {mean_tuple}")
    print(f"Std:  {std_tuple}")

    return mean_tuple, std_tuple

