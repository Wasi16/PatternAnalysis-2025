"""
train.py
--------

Contain the source code for training, validating, testing and saving ConvNeXt model. 
The model is imported from “modules.py” and the data loader is imported from “dataset.py”. 

"""
import torch
import torch.nn as nn
import torch.optim as optim 
from tqdm import tqdm # progress bars
from modules import ConvNeXt_T
import matplotlib.pyplot as plt 
from dataset import train_loader, test_loader

# Initial testing on data loading
def load_data():
    print("Start....")
    train_data, val_data, classes = train_loader(batch_size=8)

    print(f"Classes detected: {classes}")
    print(f"Training batches: {len(train_data)}")
    print(f"Validation batches: {len(val_data)}")

    # Fetch one batch from the train loader
    images, labels = next(iter(train_data))
    print(f"Batch shape: {images.shape}")  # Expect [8, 1, 256, 256]
    print(f"Labels: {labels.tolist()}")

    # Visualise
    fig, axes = plt.subplots(1, 6, figsize=(12, 3))
    for i in range(6):
        axes[i].imshow(images[i][0], cmap="gray")
        axes[i].set_title(f"{classes[labels[i]]}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

    return 0

# Configuration and setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "convnext_tiny_best.pth"

# Utility functions 
def train_one_epoch(model,dataloader, criterion, optimizer):
    """ Run one training epoch """

    model.train()
    current_loss, correct, total = 0.0, 0, 0 # running totals for loss and accuracy

    # iterate over mini batches
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images,labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad() 
        outputs = model(images) # forward pass
        loss = criterion(outputs,labels) # compute scaler loss
        loss.backward() # backpropagate
        optimizer.step() 

        current_loss += loss.item() * images.size(0)
        _,preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
    
    avg_loss = current_loss / total # mean loss across all training samples
    accuracy = 100.0 * correct / total 
    return avg_loss, accuracy

def validate(model,dataloader,criterion):
    """ Validate the model performance on calidation set"""

    model.eval() # evaluation mode
    running_loss, correct, total = 0.0, 0,0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def test(model, dataloader):
    """ Evaluate the model on test set. """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    return 100.0 * correct / total