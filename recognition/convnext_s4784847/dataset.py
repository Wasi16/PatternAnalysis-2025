"""
dataset.py 
----------
Loads and preprocess the ADNI dataset for Alzhimer's classification.

"""
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
from utils import mean_std_calc

#ADNI_DATA_PATH = "/home/groups/comp3710/ADNI"
ADNI_DATA_PATH = "C:/Wasana/Uni/sem2_2025/COMP3701/Project/Data/ADNI/AD_NC"

# Hyperparameters
IMAGE_SIZE = (256,256) 
BATCH_SIZE = 16 
VAL_SPLIT = 0.2 
SEED = 42 

def load_values():
    mean, std = mean_std_calc(os.path.join(ADNI_DATA_PATH, "train"), grayscale= True)
    return mean,std

def get_transforms(train=True, std = 0.5, mean = 0.5):
    """Return transform pipeline for train/test."""
    if train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandAugment(num_ops=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
    
def train_loader(batch_size=BATCH_SIZE, val_split=VAL_SPLIT):
    """Load ADNI training data and split into train/validation sets."""
    torch.manual_seed(SEED)
    train_dir = os.path.join(ADNI_DATA_PATH, "train")
    mean,std = load_values()

    full_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(train=True, std=std, mean=mean))
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Classes: {full_dataset.classes}")
    print(f"Train: {train_size}, Val: {val_size}")
    return train_loader, val_loader, full_dataset.classes

def test_loader(batch_size= BATCH_SIZE):
    """Load ADNI test data"""
    test_dir = os.path.join(ADNI_DATA_PATH, "test")
    mean,std = load_values()

    test_dataset = datasets.ImageFolder(test_dir, transform=get_transforms(train=False, std=std, mean=mean))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"test samples: {len(test_dataset)}")
    return test_loader
