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

MEAN, STD = mean_std_calc(os.path.join(ADNI_DATA_PATH, "train"), grayscale= True)

def get_transforms(train=True):
    """Return transform pipeline for train/test."""
    if train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandAugment(num_ops=2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
