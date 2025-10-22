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

batch_size = 16
learning_rate =  1e-4
epochs = 15

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

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    """Plots and saves loss and accuracy curves."""
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curves"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Accuracy Curves"); plt.legend()

    plt.tight_layout()
    plt.savefig("training_metrics.png", dpi=300)
    plt.show()
    print("saved training metrics")

def main():

    # Data loading
    print(" >>>> Loading data <<<< ")
    train_load, val_load, classes = train_loader(batch_size=batch_size)
    test_load = test_loader(batch_size=batch_size)

    print(f"Train : {len(train_load)}")
    print(f"Validation : {len(val_load)}")
    print(f"Test : {len(test_load)}")
    
    model = ConvNeXt_T(in_ch=1, num_classes=len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss() # Loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    
    # Main Training loop
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f"\n Epoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_one_epoch(model, train_load, criterion, optimizer)
        val_loss, val_acc = validate(model, val_load, criterion)
        scheduler.step()
        
        # Track Progress and save model 
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), SAVE_PATH)
            best_val_acc = val_acc
            print(f" Save new best model (Val Acc: {val_acc:.2f}%)")
    
    plot_metrics(train_losses, val_losses, train_accs, val_accs)

    # Final Test
    print("\n >>>> Testing best model <<<<")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    test_acc = test(model, test_load)
    print(f" Final Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()

