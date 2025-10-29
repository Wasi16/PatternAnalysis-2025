"""
predict.py
-----------
Demonstration of trained ConvNeXt-S model predictions on ADNI dataset.
"""
import torch
from modules import ConvNeXt_S

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
