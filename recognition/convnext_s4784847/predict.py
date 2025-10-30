"""
predict.py
-----------
Demonstration of trained ConvNeXt-S model predictions on ADNI dataset.
"""
import torch
from PIL import Image
import random
import matplotlib.pyplot as plt
import os
from modules import ConvNeXt_S
from dataset import get_transforms, test_loader, ADNI_DATA_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "convnext_small_best.pth"
CLASSES = ["AD", "NC"]
MEAN, STD = (0.1155,), (0.2212,) # Calculated before training

# Load Model
def load_model(model_path=MODEL_PATH, num_classes=2):
    model = ConvNeXt_S(in_ch=1, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"Model Loaded from {model_path}")
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

    # Display image
    plt.imshow(Image.open(image_path), cmap="gray")
    plt.title(f"Predicted: {CLASSES[pred_idx]} ({confidence*100:.1f}%)")
    plt.axis("off")
    plt.show()

def batch_predict_and_visualize(model, samples_per_class=8, save_path="sample_predictions.png"):
    """Run model on test set and create grid image with correct/wrong labels."""
    test_dir = os.path.join(ADNI_DATA_PATH, "test")
    transform = get_transforms(train=False, mean=MEAN, std=STD)

    # Collect equal number of images from each class
    selected_paths, true_labels = [], []
    for cls in CLASSES:
        folder = os.path.join(test_dir, cls)
        imgs = [os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(imgs)
        imgs = imgs[:samples_per_class]
        selected_paths += imgs
        true_labels += [CLASSES.index(cls)] * len(imgs)

    if not selected_paths:
        print("No test images found.")
        return

    # Transform and predict
    tensors = torch.stack([transform(Image.open(p).convert("RGB"))
                           for p in selected_paths]).to(DEVICE)
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        outputs = model(tensors)
        preds = outputs.argmax(dim=1).cpu().tolist()

    # Denormalize
    mean_t = torch.tensor(MEAN).view(1, -1, 1, 1)
    std_t = torch.tensor(STD).view(1, -1, 1, 1)
    tensors = tensors.cpu() * std_t + mean_t

    # Plot grid (balanced AD/NC)
    total = len(selected_paths)
    rows = cols = int((total) ** 0.5)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i >= total:
            ax.axis("off")
            continue
        img = tensors[i].squeeze(0).numpy()
        ax.imshow(img, cmap="gray")

        true_lbl = CLASSES[true_labels[i]]
        pred_lbl = CLASSES[preds[i]]
        correct = true_lbl == pred_lbl
        color = "green" if correct else "red"
        ax.set_title(f"P:{pred_lbl}\nT:{true_lbl}",
                     color=color, fontsize=9)
        ax.axis("off")

    plt.suptitle("Balanced Sample Predictions (Green=Correct, Red=Wrong)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved balanced prediction grid to {save_path}")

def main():
    """Run both random single prediction and balanced grid visualization."""
    model = load_model()

    #  random image from all classes 
    test_dir = os.path.join(ADNI_DATA_PATH, "test")
    all_paths = []
    for cls in CLASSES:
        folder = os.path.join(test_dir, cls)
        all_paths += [os.path.join(folder, f)
                      for f in os.listdir(folder)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not all_paths:
        print(" No test images found.")
        return
    
    # Predict a single image
    random_img = random.choice(all_paths)
    predict_single_image(model, random_img)

    # balanced prediction grid
    batch_predict_and_visualize(model)

if __name__ == "__main__":
    main()
