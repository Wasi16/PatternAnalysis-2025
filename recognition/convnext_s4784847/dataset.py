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
from collections import defaultdict
from sklearn.model_selection import train_test_split

#ADNI_DATA_PATH = "/home/groups/comp3710/ADNI"
ADNI_DATA_PATH = "C:/Wasana/Uni/sem2_2025/COMP3701/Project/Data/ADNI/AD_NC"

# Hyperparameters
IMAGE_SIZE = (256,256) 
BATCH_SIZE = 16 
VAL_SPLIT = 0.2 
SEED = 42 

def extract_patient_id(filename):
    """
    Extract patient ID from filename.
    Example: '388206_78.jpeg' -> '388206'
    """
    return filename.split('_')[0]

def get_patient_splits(data_dir, val_split=0.2, seed=42):
    """
    Split data by patient ID to prevent data leakage.
    Returns train and val patient IDs for each class.
    """
    patient_to_class = {}
    
    # Iterate through class folders (AD and NC)
    for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        # Get all patient IDs in this class
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

class ADNIDataset(Dataset):
    """
    Custom ADNI Dataset that tracks patient IDs.
    """
    def __init__(self, root_dir, transform=None, patient_ids=None):
        """
        Args:
            root_dir: Directory with class subdirectories
            transform: Optional transform to apply
            patient_ids: Set of patient IDs to include (None = all)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.patient_ids = patient_ids
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load samples
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            class_idx = self.class_to_idx[class_name]
            for filename in os.listdir(class_path):
                if not filename.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                patient_id = extract_patient_id(filename)
                
                # Filter by patient_ids if provided
                if patient_ids is None or patient_id in patient_ids:
                    filepath = os.path.join(class_path, filename)
                    self.samples.append((filepath, class_idx, patient_id))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, label, patient_id = self.samples[idx]
        image = Image.open(filepath).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, patient_id


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
