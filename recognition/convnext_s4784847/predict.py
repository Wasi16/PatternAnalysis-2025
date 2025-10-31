"""
predict.py
-----------
Demonstration of trained ConvNeXt-S model predictions on the ADNI dataset.

This script performs:
1. Single image prediction (visualized with confidence score)
2. Batch prediction on multiple images (balanced across classes)
   with a grid visualization showing correct and incorrect predictions.
"""
import os
import random
import matplotlib.pyplot as plt
import torch
from dataset import ADNI_DATA_PATH, get_transforms
from modules import ConvNeXt_S
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "convnext_small_best.pth"   # Path to best model checkpoint
CLASSES = ["AD", "NC"]                   # Alzheimer’s vs Normal Controls
MEAN, STD = (0.1155,), (0.2212,)         # Normalization values (pre-computed)

def load_model(model_path=MODEL_PATH, num_classes=2):
    """
    Load the trained ConvNeXt-S model from checkpoint.

    Args:
        model_path (str): Path to model file (.pth)
        num_classes (int): Number of output classes (default = 2)
    
    Returns:
        model (torch.nn.Module): Loaded ConvNeXt-S model in evaluation mode.
    """
    model = ConvNeXt_S(in_ch=1, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"Model Loaded from {model_path}")
    return model

def predict_single_image(model, image_path, mean=(0.1155,), std=(0.2212,)):
    """
    Predict Alzheimer’s vs Normal Control for a single MRI image.

    Args:
        model: Trained model for inference.
        image_path (str): Path to image file.
        mean, std (tuple): Normalization parameters.
    """
    # Apply same preprocessing as during training
    transform = get_transforms(train=False, mean=mean, std=std)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    # Perform inference with mixed precision (for efficiency)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    # Print and visualize prediction
    print(f"\nPrediction for {os.path.basename(image_path)}:")
    print(f"→ {CLASSES[pred_idx]} ({confidence*100:.2f}% confidence)")

    # Display the image with predicted label
    plt.imshow(Image.open(image_path), cmap="gray")
    plt.title(f"Predicted: {CLASSES[pred_idx]} ({confidence*100:.1f}%)")
    plt.axis("off")
    plt.show()

def batch_predict_and_visualize(model, samples_per_class=8, save_path="sample_predictions.png"):
    """
    Run model on multiple images (balanced across both classes) and visualize results.

    Args:
        model: Trained ConvNeXt-S model.
        samples_per_class (int): Number of test images per class to visualize.
        save_path (str): File path to save output grid image.
    """
    test_dir = os.path.join(ADNI_DATA_PATH, "test")
    transform = get_transforms(train=False, mean=MEAN, std=STD)

    # Collect an equal number of images from each class (balanced sampling)
    selected_paths, true_labels = [], []
    for cls in CLASSES:
        folder = os.path.join(test_dir, cls)
        imgs = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(imgs)
        imgs = imgs[:samples_per_class]
        selected_paths += imgs
        true_labels += [CLASSES.index(cls)] * len(imgs)

    if not selected_paths:
        print("No test images found.")
        return

    # Apply preprocessing and stack all images into a single tensor batch
    tensors = torch.stack([
        transform(Image.open(p).convert("RGB")) for p in selected_paths
    ]).to(DEVICE)

    # Perform inference on all selected images
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        outputs = model(tensors)
        preds = outputs.argmax(dim=1).cpu().tolist()

    # convert back to visible pixel range
    mean_t = torch.tensor(MEAN).view(1, -1, 1, 1)
    std_t = torch.tensor(STD).view(1, -1, 1, 1)
    tensors = tensors.cpu() * std_t + mean_t

    # Create a grid visualization
    total = len(selected_paths)
    rows = cols = int(total ** 0.5)  # create a square grid
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        if i >= total:
            ax.axis("off")
            continue

        img = tensors[i].squeeze(0).numpy()  # remove channel dimension
        ax.imshow(img, cmap="gray")

        true_lbl = CLASSES[true_labels[i]]
        pred_lbl = CLASSES[preds[i]]
        correct = true_lbl == pred_lbl
        color = "green" if correct else "red"
        ax.set_title(f"P:{pred_lbl}\nT:{true_lbl}", color=color, fontsize=9)
        ax.axis("off")

    plt.suptitle("Balanced Sample Predictions (Green=Correct, Red=Wrong)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved balanced prediction grid to {save_path}")

def main():
    """
    1. Loads model checkpoint.
    2. Randomly selects one image for single prediction.
    3. Runs batch prediction to visualize results across both classes.
    """
    model = load_model()

    # Gather all available test images
    test_dir = os.path.join(ADNI_DATA_PATH, "test")
    all_paths = []
    for cls in CLASSES:
        folder = os.path.join(test_dir, cls)
        all_paths += [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    if not all_paths:
        print("No test images found.")
        return

    # Single image prediction
    random_img = random.choice(all_paths)
    predict_single_image(model, random_img)

    # Balanced Batch Visualization
    batch_predict_and_visualize(model)

if __name__ == "__main__":
    main()
