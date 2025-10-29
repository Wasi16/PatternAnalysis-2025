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
from modules import ConvNeXt_S
import matplotlib.pyplot as plt 
from dataset import train_loader, test_loader
from sklearn.metrics import f1_score, confusion_matrix
import itertools
import numpy as np
import wandb

# Configuration and setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "convnext_small_best.pth"

batch_size = 12 #32
learning_rate = 2e-4  #3e-4
epochs = 100 #150

# Utility functions 
def train_one_epoch(model,dataloader, criterion, optimizer,scaler):
    """ Run one training epoch """

    model.train()
    current_loss, correct, total = 0.0, 0, 0 # running totals for loss and accuracy

    # iterate over mini batches
    for batch in tqdm(dataloader, desc="Training", leave=False):
        # Handle both dataset types
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch
        
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        # Mixed Preicision
        with torch.amp.autocast("cuda"):
            outputs = model(images) # forward pass
            loss = criterion(outputs,labels) # compute scaler loss

        scaler.scale(loss).backward() # backpropagate
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        current_loss += loss.item() * images.size(0)
        _,preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
    
    avg_loss = current_loss / total # mean loss across all training samples
    accuracy = 100.0 * correct / total 
    return avg_loss, accuracy

def validate(model,dataloader,criterion, classes):
    """ Validate the model performance on calidation set"""

    model.eval() # evaluation mode
    running_loss, correct, total = 0.0, 0,0
    all_preds, all_labels = [],[]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            # Handle both dataset types
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")

    
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))
    cm_path = plot_confusion_matrix(cm, classes, normalize=True, out_path="val_confusion.png")

    wandb.log({"val/confusion_matrix": wandb.Image(cm_path)})


    return avg_loss, accuracy, f1

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

def test_with_aggregation(model, dataloader, use_aggregation=True):
    """
    Test with optional multi-scan aggregation per patient.
    
    If use_aggregation=True:
        - Average predictions across all scans for each patient
        - More robust but requires patient-grouped test loader
    """
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", leave=False):
            if use_aggregation and len(batch) == 3:
                # Patient-level aggregation
                images, labels, num_scans = batch
                images = images.squeeze(0)  # Remove batch dimension
                labels = labels.to(DEVICE)
                
                # Get predictions for all scans
                scan_outputs = []
                for i in range(0, len(images), 16):  # Process in mini-batches
                    batch_scans = images[i:i+16].to(DEVICE)
                    with torch.amp.autocast("cuda"):
                        outputs = model(batch_scans)
                    scan_outputs.append(outputs)
                
                # Average logits across all scans
                all_outputs = torch.cat(scan_outputs, dim=0)
                avg_output = all_outputs.mean(dim=0, keepdim=True)
                _, pred = avg_output.max(1)
                
                total += 1
                correct += pred.eq(labels).sum().item()
                all_preds.append(pred.cpu().item())
                all_labels.append(labels.cpu().item())
            else:
                # Standard single-scan prediction
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                    
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                _, preds = outputs.max(1)
                
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")
    
    return accuracy, f1, all_preds, all_labels

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

def plot_confusion_matrix(cm, classes, normalize=True, title="Confusion matrix", out_path="conf_matrix.png"):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=8)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path

def main():
        
    wandb.init(
    project="convnext-adni",    
    name="convnext_small_run",   
    config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "AdamW",
        "scheduler": "StepLR",
        "scheduler": "CosineAnnealingLR",
        "architecture": "ConvNeXt-Tiny",
        "label_smoothing": 0.1,
        "weight_decay": 0.03
        },
        save_code = True
    )

    # Data loading
    print(" >>>> Loading data <<<< ")
    train_load, val_load, classes = train_loader(batch_size=batch_size,use_patient_grouping=True )
    test_load_single = test_loader(batch_size=batch_size, use_patient_aggregation=False)
    test_load_agg = test_loader(batch_size=1, use_patient_aggregation=True)

    print(f"Train : {len(train_load)}")
    print(f"Validation : {len(val_load)}")
    print(f"Test : {len(test_load_single)}")
    print(f"classes: {classes}")
    
    model = ConvNeXt_S(in_ch=1, num_classes=len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss() # Loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")

    wandb.watch(model, criterion, log="all", log_freq=100)
    
    # Main Training loop
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = 8
    early_stop = False

    for epoch in range(epochs):
        print(f"\n Epoch {epoch+1}/{epochs}")
        
        # Training
        train_loss, train_acc = train_one_epoch(model, train_load, criterion, optimizer, scaler)
        # Validation
        val_loss, val_acc, val_f1 = validate(model, val_load, criterion, classes)
        scheduler.step()
        
        # Track Progress and save model 
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.3f}")
        
        # Log metrics to weights and biases
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/f1": val_f1,
            "lr": scheduler.get_last_lr()[0]
        })

        # Early stopping logic (based on validation loss) 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            epochs_no_improve = 0
            print(f"Validation loss improved to {val_loss:.4f}. Model saved.")
        else:
            epochs_no_improve += 1
            print(f" No improvement for {epochs_no_improve} epoch(s).")

        # Stop if no improvement for 'patience' epochs
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            wandb.log({"early_stop_epoch": epoch + 1})
            early_stop = True
            break

    # Plot training
    plot_metrics(train_losses, val_losses, train_accs, val_accs)

    if early_stop:
        print("Loading best saved model before early stop...")
    else:
        print("Training completed full schedule. Loading best model...")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))

    # Final Test
    print("\n >>>> Testing best model <<<<")
    test_acc = test(model, test_load)
    print(f" Final Test Accuracy: {test_acc:.2f}%")

    wandb.log({"test/acc": test_acc})

    wandb.finish()

if __name__ == "__main__":
    main()

