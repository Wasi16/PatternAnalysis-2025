import torch
import matplotlib.pyplot as plt 
from dataset import train_loader, test_loader

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

if __name__ == "__main__":
    load_data()