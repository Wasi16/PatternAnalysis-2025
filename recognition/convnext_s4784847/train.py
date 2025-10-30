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
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import itertools
import numpy as np
import wandb

# Configuration and setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "convnext_small_best.pth"

batch_size = 28
learning_rate = 1e-4
epochs = 65

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
        "scheduler": "ReduceLROnPlateau",
        "architecture": "ConvNeXt-Small",
        "label_smoothing": 0.1,
        "weight_decay": 0.03
        },
        save_code = True
    )

    # Data loading
    print(" >>>> Loading data <<<< ")
    train_load, val_load, classes = train_loader(batch_size=batch_size,use_patient_grouping=False )
    test_load_single = test_loader(batch_size=batch_size, use_patient_aggregation=False)
    test_load_agg = test_loader(batch_size=1, use_patient_aggregation=True)

    print(f"Train : {len(train_load)}")
    print(f"Validation : {len(val_load)}")
    print(f"Test : {len(test_load_single)}")
    print(f"classes: {classes}")
    
    model = ConvNeXt_S(in_ch=1, num_classes=len(classes)).to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n>>> Calculating class weights from training dataset...")
    class_counts = [0, 0]
    for _, label, *_ in train_load.dataset:
        if isinstance(label, torch.Tensor):
            label = int(label.item()) 
        class_counts[label] += 1
    
    w_AD = 1.65
    w_NC = 1.35

    print(f"Class counts → AD: {class_counts[0]}, NC: {class_counts[1]}")
    print(f"Class weights → AD: {w_AD:.3f}, NC: {w_NC:.3f}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=torch.tensor([w_AD,w_NC], dtype=torch.float32).to(DEVICE)) # Loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05,betas=(0.9, 0.999) )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6, min_lr=1e-7)

    scaler = torch.amp.GradScaler("cuda")

    wandb.watch(model, criterion, log="all", log_freq=100)
    
    # Main Training loop
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        print(f"\n Epoch {epoch+1}/{epochs}")
        
        # Training
        train_loss, train_acc = train_one_epoch(model, train_load, criterion, optimizer, scaler)
        # Validation
        val_loss, val_acc, val_f1 = validate(model, val_load, criterion, classes)
        scheduler.step(val_f1)
        current_lr = scheduler.get_last_lr()[0]
        
        # Track Progress and save model 
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.3f}")
        print(f"LR: {current_lr:.2e}")

        # Log metrics to weights and biases
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/f1": val_f1,
            "lr": current_lr
        })

        # Save best model based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
            }, SAVE_PATH)
            epochs_no_improve = 0
            print(f"New best F1: {val_f1:.3f} (Acc: {val_acc:.2f}%)")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")

    # Plot training
    plot_metrics(train_losses, val_losses, train_accs, val_accs)

    print("Training completed. Loading best model...")
    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_acc_single, test_f1_single, preds, labels = test_with_aggregation(
        model, test_load_single, use_aggregation=False
    )
    print(f"Test Accuracy: {test_acc_single:.2f}%")
    print(f"Test F1 Score: {test_f1_single:.3f}")
    
    # Test with multi-scan aggregation
    print(f"\n{'='*60}")
    print(f" Testing (Multi-Scan Aggregation)...")
    print(f"{'='*60}")
    
    test_acc_agg, test_f1_agg, preds_agg, labels_agg = test_with_aggregation(
        model, test_load_agg, use_aggregation=True
    )
    print(f"Test Accuracy: {test_acc_agg:.2f}%")
    print(f"Test F1 Score: {test_f1_agg:.3f}")
    
    # Confusion matrix for aggregated test
    test_cm = confusion_matrix(labels_agg, preds_agg)
    plot_confusion_matrix(test_cm, classes, normalize=True, 
                         title="Test Confusion Matrix (Aggregated)",
                         out_path="test_confusion_agg.png")
    
    # Classification report
    print(f"\n{'='*60}")
    print(" Classification Report (Aggregated):")
    print(f"{'='*60}")
    print(classification_report(labels_agg, preds_agg, target_names=classes))
    
    # Log final results
    wandb.log({
        "test/acc_single": test_acc_single,
        "test/f1_single": test_f1_single,
        "test/acc_aggregated": test_acc_agg,
        "test/f1_aggregated": test_f1_agg,
        "test/confusion_matrix": wandb.Image("test_confusion_agg.png")
    })

    wandb.log({"test/acc": test_acc_single})
    wandb.finish()

if __name__ == "__main__":
    main()

