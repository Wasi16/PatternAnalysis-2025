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