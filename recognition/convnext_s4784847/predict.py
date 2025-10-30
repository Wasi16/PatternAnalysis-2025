"""
predict.py
-----------
Demonstration of trained ConvNeXt-S model predictions on ADNI dataset.
"""
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
from modules import ConvNeXt_S
from dataset import get_transforms, test_loader, ADNI_DATA_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "convnext_small_best.pth"
CLASSES = ["AD", "NC"]

# Load Model
def load_model(model_path=MODEL_PATH, num_classes=2):
    model = ConvNeXt_S(in_ch=1, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"Loaded model from {model_path}")
    return model

def predict_single_image(model, image_path, mean=(0.1155,), std=(0.2212,)):
    """Predict Alzheimer’s vs Normal for a single image."""
    transform = get_transforms(train=False, mean=mean, std=std)
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    print(f"\n Prediction for {os.path.basename(image_path)}:")
    print(f"Predicted class: {CLASSES[pred_idx]} ({confidence*100:.2f}% confidence)")

    # Display image
    plt.imshow(Image.open(image_path), cmap="gray")
    plt.title(f"Predicted: {CLASSES[pred_idx]} ({confidence*100:.1f}%)")
    plt.axis("off")
    plt.show()

def batch_predict_and_visualize(model, save_path="sample_predictions.png", num_samples=16):
    """Run model on test set and create grid image with correct/wrong labels."""
    loader = test_loader(batch_size=8, use_patient_aggregation=False)
    mean, std = (0.1155,), (0.2212,)
    transform = get_transforms(train=False, mean=mean, std=std)

    all_images, all_preds, all_labels = [], [], []
    model.eval()

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for images, labels, *_ in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_images.append(images.cpu())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            if len(all_preds) >= num_samples:
                break

    # Concatenate first few batches
    all_images = torch.cat(all_images, dim=0)[:num_samples]
    all_preds = all_preds[:num_samples]
    all_labels = all_labels[:num_samples]

    # Denormalize for display
    mean_t = torch.tensor(mean).view(1, -1, 1, 1)
    std_t = torch.tensor(std).view(1, -1, 1, 1)
    all_images = all_images * std_t + mean_t

    # Create figure
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i >= len(all_images):
            ax.axis("off")
            continue
        img = all_images[i].squeeze(0).numpy()
        ax.imshow(img, cmap="gray")

        true_label = CLASSES[all_labels[i]]
        pred_label = CLASSES[all_preds[i]]
        correct = all_labels[i] == all_preds[i]
        color = "green" if correct else "red"
        ax.set_title(f"P:{pred_label}\nT:{true_label}",
                     color=color, fontsize=9)
        ax.axis("off")

    plt.suptitle("Sample Predictions (Green=Correct, Red=Wrong)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved prediction grid to {save_path}")