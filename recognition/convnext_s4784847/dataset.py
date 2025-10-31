"""
dataset.py 
----------
Loads and preprocesses the ADNI dataset for Alzheimer’s classification.
Includes patient-level grouping to prevent data leakage between splits.
"""

import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils import mean_std_calc

# Path to ADNI dataset directory (with subfolders AD/NC and train/test)
ADNI_DATA_PATH = "C:/Wasana/Uni/sem2_2025/COMP3701/Project/Data/ADNI/AD_NC"

# General hyperparameters for data handling
IMAGE_SIZE = (256, 256)   # Image resizing resolution
BATCH_SIZE = 16           # Batch size for loaders
VAL_SPLIT = 0.2           # Validation split ratio (20%)
SEED = 42                 # Random seed for reproducibility


# Utility Functions
def extract_patient_id(filename):
    """
    Extracts patient ID from a filename.
    Example: '388206_78.jpeg' -> '388206'
    
    Each scan filename contains the patient ID as a prefix before the underscore.
    This allows grouping scans belonging to the same individual.
    """
    return filename.split('_')[0]


def get_patient_splits(data_dir, val_split=0.2, seed=42):
    """
    Splits patients into training and validation sets to prevent data leakage.
    Each patient’s scans stay entirely in one set.

    Args:
        data_dir (str): Path to the training directory (contains AD and NC folders)
        val_split (float): Fraction of patients to include in validation set
        seed (int): Random seed for reproducibility
    
    Returns:
        (set, set): Train and validation patient ID sets
    """
    patient_to_class = {}
    
    # Read both class folders (AD, NC)
    for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        # Extract patient IDs from filenames
        for filename in os.listdir(class_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                patient_id = extract_patient_id(filename)
                if patient_id not in patient_to_class:
                    patient_to_class[patient_id] = class_idx
    
    # Group patients by class
    class_patients = defaultdict(list)
    for patient_id, class_idx in patient_to_class.items():
        class_patients[class_idx].append(patient_id)
    
    # Split each class separately to maintain class balance
    train_patients, val_patients = [], []
    for class_idx, patients in class_patients.items():
        train_pts, val_pts = train_test_split(
            patients, 
            test_size=val_split, 
            random_state=seed,
            shuffle=True
        )
        train_patients.extend(train_pts)
        val_patients.extend(val_pts)
    
    print(f"Train patients: {len(train_patients)}, Val patients: {len(val_patients)}")
    return set(train_patients), set(val_patients)

# Dataset Classes
class ADNIDataset(Dataset):
    """
    Standard dataset for loading individual MRI scan images.
    Used when patient-level grouping is not required.
    """

    def __init__(self, root_dir, transform=None, patient_ids=None):
        """
        Args:
            root_dir (str): Root directory containing AD/NC subfolders
            transform (callable): Optional torchvision transform
            patient_ids (set): Optional set of patient IDs to filter samples
        """
        self.root_dir = root_dir
        self.transform = transform
        self.patient_ids = patient_ids
        self.samples = []

        # Map class names to integer labels
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths and corresponding labels
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            class_idx = self.class_to_idx[class_name]
            for filename in os.listdir(class_path):
                if not filename.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                patient_id = extract_patient_id(filename)
                
                # Include only patients belonging to the provided set
                if patient_ids is None or patient_id in patient_ids:
                    filepath = os.path.join(class_path, filename)
                    self.samples.append((filepath, class_idx, patient_id))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Loads an image and applies transformations.
        Returns: (image_tensor, label, patient_id)
        """
        filepath, label, patient_id = self.samples[idx]
        image = Image.open(filepath).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, patient_id


class ADNIPatientDataset(Dataset):
    """
    Dataset that groups scans by patient.
    Supports:
        - 'single': returns one random scan per patient (for training)
        - 'all': returns all scans (for patient-level testing/aggregation)
    """

    def __init__(self, root_dir, transform=None, patient_ids=None, mode='single'):
        """
        Args:
            root_dir (str): Directory containing AD/NC subfolders
            transform (callable): Transform pipeline to apply to images
            patient_ids (set): Optional subset of patient IDs
            mode (str): 'single' or 'all'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.patient_ids = patient_ids
        self.mode = mode
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.patient_scans = defaultdict(list)
        self.patient_labels = {}

        # Group all scans belonging to each patient
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            class_idx = self.class_to_idx[class_name]
            for filename in os.listdir(class_path):
                if not filename.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                patient_id = extract_patient_id(filename)
                
                if patient_ids is None or patient_id in patient_ids:
                    filepath = os.path.join(class_path, filename)
                    self.patient_scans[patient_id].append(filepath)
                    self.patient_labels[patient_id] = class_idx
        
        self.patient_list = list(self.patient_scans.keys())
    
    def __len__(self):
        return len(self.patient_list)
    
    def __getitem__(self, idx):
        """
        Returns patient-level data:
            - mode='single': one random scan
            - mode='all': all scans as a tensor batch
        """
        patient_id = self.patient_list[idx]
        scans = self.patient_scans[patient_id]
        label = self.patient_labels[patient_id]
        
        if self.mode == 'single':
            # Randomly pick one scan for this patient
            scan_path = np.random.choice(scans)
            image = Image.open(scan_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        
        else:  # mode == 'all'
            # Return all scans for aggregation
            images = []
            for scan_path in scans:
                image = Image.open(scan_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)

            # Stack into tensor [N, C, H, W]
            images = torch.stack(images)
            return images, label, len(scans)

# Data Loader Function
def load_values():
    """
    Computes mean and standard deviation of the dataset (for normalization).
    Uses grayscale channel statistics for MRI images.
    """
    mean, std = mean_std_calc(os.path.join(ADNI_DATA_PATH, "train"), grayscale=True)
    return mean, std


def get_transforms(train=True, std=0.5, mean=0.5):
    """
    Defines image preprocessing and augmentation pipeline.
    
    Training mode includes strong augmentations for robustness.
    Testing mode uses only normalization.
    """
    if train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


def train_loader(batch_size=BATCH_SIZE, val_split=VAL_SPLIT, use_patient_grouping=True):
    """
    Creates DataLoaders for training and validation sets.

    Args:
        batch_size (int): Number of samples per batch
        val_split (float): Fraction for validation split
        use_patient_grouping (bool): Whether to sample one scan per patient per epoch
    """
    torch.manual_seed(SEED)
    train_dir = os.path.join(ADNI_DATA_PATH, "train")
    mean, std = load_values()

    # Split patients between train/validation
    train_patients, val_patients = get_patient_splits(train_dir, val_split, SEED)
    
    # Patient-grouped dataset (more robust)
    if use_patient_grouping:
        train_dataset = ADNIPatientDataset(
            train_dir, 
            transform=get_transforms(train=True, std=std, mean=mean),
            patient_ids=train_patients,
            mode='single'
        )
        val_dataset = ADNIPatientDataset(
            train_dir,
            transform=get_transforms(train=False, std=std, mean=mean),
            patient_ids=val_patients,
            mode='single'
        )
    else:
        # All scans treated as independent samples
        train_dataset = ADNIDataset(
            train_dir,
            transform=get_transforms(train=True, std=std, mean=mean),
            patient_ids=train_patients
        )
        val_dataset = ADNIDataset(
            train_dir,
            transform=get_transforms(train=False, std=std, mean=mean),
            patient_ids=val_patients
        )
    
    # Create PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    classes = train_dataset.classes
    print(f"Classes: {classes}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader, classes


def test_loader(batch_size=BATCH_SIZE, use_patient_aggregation=False):
    """
    Creates test DataLoader for final evaluation.
    
    Args:
        batch_size (int): Batch size (1 if using patient aggregation)
        use_patient_aggregation (bool): Whether to return all scans per patient
    
    Returns:
        torch.utils.data.DataLoader: Test data loader
    """
    test_dir = os.path.join(ADNI_DATA_PATH, "test")
    mean, std = load_values()

    if use_patient_aggregation:
        # Return all scans per patient for averaged prediction
        test_dataset = ADNIPatientDataset(
            test_dir,
            transform=get_transforms(train=False, std=std, mean=mean),
            patient_ids=None,
            mode='all'
        )
    else:
        # Return one random scan per patient
        test_dataset = ADNIPatientDataset(
            test_dir,
            transform=get_transforms(train=False, std=std, mean=mean),
            patient_ids=None,
            mode='single'
        )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size if not use_patient_aggregation else 1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    print(f"Test samples: {len(test_dataset)}")
    return test_loader
