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

class ADNIPatientDataset(Dataset):
    """
    Dataset that groups all scans per patient.
    Can return either all scans or a random scan per patient.
    """
    def __init__(self, root_dir, transform=None, patient_ids=None, mode='single'):
        """
        Args:
            mode: 'single' - return one random scan per patient
                  'all' - return all scans per patient (for aggregation)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.patient_ids = patient_ids
        self.mode = mode
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Group samples by patient
        self.patient_scans = defaultdict(list)
        self.patient_labels = {}
        
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
        patient_id = self.patient_list[idx]
        scans = self.patient_scans[patient_id]
        label = self.patient_labels[patient_id]
        
        if self.mode == 'single':
            # Return one random scan
            scan_path = np.random.choice(scans)
            image = Image.open(scan_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        
        else:  # mode == 'all'
            # Return all scans for this patient
            images = []
            for scan_path in scans:
                image = Image.open(scan_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            return torch.stack(images), label, len(images)

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
            transforms.Normalize(mean,std)
        ])
    
def train_loader(batch_size=BATCH_SIZE, val_split=VAL_SPLIT, use_patient_grouping=True):
    """Load ADNI training data and split into train/validation sets."""
    torch.manual_seed(SEED)
    train_dir = os.path.join(ADNI_DATA_PATH, "train")
    mean,std = load_values()

    # Get patient-level splits
    train_patients, val_patients = get_patient_splits(train_dir, val_split, SEED)
    
    if use_patient_grouping:
        # One random scan per patient per epoch
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
        # All scans as separate samples
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,pin_memory=True)
    
    classes = train_dataset.classes
    print(f"Classes: {classes}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader, classes

def test_loader(batch_size=BATCH_SIZE, use_patient_aggregation=False):
    """
    Load ADNI test data.
    """
    test_dir = os.path.join(ADNI_DATA_PATH, "test")
    mean, std = load_values()

    if use_patient_aggregation:
        test_dataset = ADNIPatientDataset(
            test_dir,
            transform=get_transforms(train=False, std=std, mean=mean),
            patient_ids=None,
            mode='all'
        )
    else:
        test_dataset = ADNIDataset(
            test_dir,
            transform=get_transforms(train=False, std=std, mean=mean),
            patient_ids=None
        )

    test_loader = DataLoader(test_dataset, batch_size=batch_size if not use_patient_aggregation else 1, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Test samples: {len(test_dataset)}")
    return test_loader
