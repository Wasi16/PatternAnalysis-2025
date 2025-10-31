# Alzheimer's Disease Classification using ConvNeXt-Small

**Name:** Wasana Nawamini Udabage  
**Student No:** 47848476

## Overview

This repository implements a ConvNeXt-Small model for classifying Alzheimer's Disease (AD) vs. Cognitively Normal (NC) patients using MRI scans from the ADNI dataset. Built for COMP3710 Pattern Recognition project 8, this implementation features patient-level data splitting to prevent leakage, robust data augmentation, class balancing, and multi scan prediction aggregation. The project achieves 75% test accuracy, below the required 0.8 threshold, with full experiment tracking via Weights & Biases.

## Table of Contents

1. [Overview](#overview)
2. [ConvNeXt Overview](#convnext-overview)
3. [Problem Space](#problem-space)
4. [Project Structure](#project-structure)
5. [Model Architecture](#model-architecture)
6. [Replication of Results](#replication-of-results)
7. [Data Loading](#data-loading)
8. [Training](#training)
9. [Testing](#testing)
10. [Results](#results)
11. [Discussion](#discussion)
12. [Future Improvements](#future-improvements)
13. [Conclusion](#conclusion)
14. [References](#references)

---

## ConvNeXt Overview

ConvNeXt is a modern interpretation of convolutional neural networks(ConvNets), often referred to as a "ConvNet for the 2020s". It was introduced by Liu et al.(2022) as a way to bring traditional ConvNets up to the same level of performance as Vision Transformers, while preserving the efficiency and simplicity of convolutional models.

The model combines design concepts from ResNet and Vision Transformers, taking the best from both. From ResNet, ConvNeXt keeps the general stage based architecture and residual connections. From Vision Transformers, it adopts concepts such as patchify stem, large receptive fields, layer normalisation and GELU activations.

The major architectural updates for ConvNext include:

- **Patchify Stem:** Replacing the traditional ResNet stem with a 4×4 convolution (stride 4), similar to how Vision Transformers patchify
- **Large Depthwise Convolutions:** Using 7×7 depthwise convolutions inside each block to capture a larger spatial context
- **Inverted Bottleneck Structure:** Expanding and projecting channels using an inverted bottleneck
- **Modern Normalization and Activation:** Switching from BatchNorm + ReLU to LayerNorm + GELU for smoother and more stable learning
- **Simplified Design:** Four clear stages with downsampling between each one, creating a hierarchical feature representation

### Why is it suitable for Medical Imaging?

I selected ConvNeXt-Small for this task because its design aligns well with the challenges of medical image analysis, particularly for MRI-based Alzheimer's disease classification:

1. **Large Receptive Fields:** The 7×7 depthwise convolutions enable the model to capture broad spatial relationships across brain regions, which is crucial for detecting subtle structural changes associated with Alzheimer's disease such as hippocampal atrophy and ventricular enlargement.

2. **Stable Training on Grayscale Data:** The use of Layer Normalization and GELU activation helps maintain stable training on grayscale MRI data, which typically have lower contrast and higher variability than natural RGB images.

3. **Hierarchical Feature Extraction:** The four-stage architecture allows ConvNeXt to extract both local texture details (early stages) and global anatomical patterns (later stages), improving classification robustness across different brain regions.

4. **Computational Efficiency:** ConvNeXt combines the simplicity and efficiency of convolutional networks with transformer-inspired design principles, offering strong performance while remaining computationally tractable for medical imaging tasks.

---

## Problem Space

The focus of this project is on the classification of Alzheimer's disease using MRI scans from the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset. Alzheimer's causes gradual structural changes in the brain, and early detection is crucial for treatment and diagnosis. The goal is to build a model capable of distinguishing between Alzheimer's patients (AD) and cognitively normal controls (NC) based on 2D MRI slices. This task represents a real world medical image classification challenge, where accurate pattern recognition from subtle anatomical differences is essential.

---

## Project Structure

The implementation is structured into five main files for clarity and maintainability:

- **dataset.py:** Handles ADNI data loading, augmentation, and patient-level splitting.
- **modules.py:** Defines the ConvNeXt-Small model with ConvNeXtBlock and DownsampleLayer.
- **train.py:** Contains the training, validation, and testing loops with W&B logging.
- **utils.py:** Provides data normalization utilities (mean and standard deviation computation).
- **predict.py:** Inference and visualization.

---

## Model Architecture

The model used in this project is **ConvNeXt-Small**, a modernized convolutional neural network designed to combine the strengths of classic ConvNets like ResNet with concepts from Vision Transformers.
It contains four hierarchical stages with progressively increasing channel depth and decreasing spatial resolution.

### Architecture Overview

ConvNeXt-Small consists of 36 total convolutional blocks organized across four stages:

**Stage Configuration:**

- **Stage 1:** 3 ConvNeXt blocks, 96 channels, spatial size: 64×64
- **Stage 2:** 3 ConvNeXt blocks, 192 channels, spatial size: 32×32
- **Stage 3:** 27 ConvNeXt blocks, 384 channels, spatial size: 16×16
- **Stage 4:** 3 ConvNeXt blocks, 768 channels, spatial size: 8×8

```
Input [1×256×256]
    ↓
┌─────────────────────────┐
│  Patchify Stem (4×4)    │  → [96×64×64]
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Stage 1 (3 blocks)     │  → [96×64×64]
│  ├─ ConvNeXt Block      │
│  ├─ ConvNeXt Block      │
│  └─ ConvNeXt Block      │
└─────────────────────────┘
    ↓ [Downsample 2×2]
┌─────────────────────────┐
│  Stage 2 (3 blocks)     │  → [192×32×32]
└─────────────────────────┘
    ↓ [Downsample 2×2]
┌─────────────────────────┐
│  Stage 3 (27 blocks)    │  → [384×16×16]  ← Deepest stage
└─────────────────────────┘
    ↓ [Downsample 2×2]
┌─────────────────────────┐
│  Stage 4 (3 blocks)     │  → [768×8×8]
└─────────────────────────┘
    ↓ [Global Avg Pool]
┌─────────────────────────┐
│  Classification Head    │
│  ├─ LayerNorm           │
│  ├─ Dropout (0.3)       │
│  └─ Linear(768→2)       │
└─────────────────────────┘
    ↓
Output [2] (AD, NC logits)
```

### Key Components

#### 1. Patchify Stem

The patchify stem serves as the network's input embedding layer, converting raw MRI images into a feature rich representation. Rather than processing pixels individually, it divides the 256×256 input image into non overlapping 4×4 patches and projects each patch into a 96-dimensional feature vector. This aggressive spatial downsampling (16× reduction) is inspired by Vision Transformers and enables the network to focus on meaningful image structures rather than low-level pixel details, while simultaneously reducing computational cost for subsequent layers.

```python
nn.Conv2d(in_channels=1, out_channels=96, kernel_size=4, stride=4)
nn.LayerNorm(96)
```

#### 2. ConvNeXt Block

The ConvNeXt block is the main building unit that processes features at each stage of the network. With an inverted bottleneck design where features are first processed through a large 7×7 depthwise convolution to capture spatial context, then expanded to 4× the channel dimensions for rich feature learning, and finally projected back to the original size through a residual connection. This design allows the network to learn both local texture patterns and broader spatial relationships while maintaining computational efficiency.

Each block contains:

- **Depthwise Convolution (7×7):** Captures large spatial context with fewer parameters than standard convolution
- **Layer Normalization:** Stabilizes training and normalizes features per channel
- **Pointwise Expansion (1×1 Linear):** Expands channels by 4× for increased capacity
- **GELU Activation:** Smooth, non saturating activation function for better gradient flow
- **Pointwise Projection (1×1 Linear):** Projects back to original dimensions
- **Residual Connection:** Enables gradient flow and facilitates training of deep networks

![ConvNeXt Block](README%20resources/image.png)

#### 3. Downsampling Layer

Downsampling layers connect consecutive stages using 2×2 convolutions with stride 2 to halve spatial resolution while doubling channel depth. This builds a hierarchical feature pyramid where early layers capture fine local brain details, while later layers extract broader structural patterns, mirroring ResNet's multi-scale design but within ConvNeXt's modern framework.

#### 4. Classification Head

After four stages extract hierarchical features (768 channels, 8×8 spatial size), global average pooling condenses them into a single 768-dimensional vector representing the whole brain. This vector passes through LayerNorm for stability, 30% dropout for regularization, and a final linear layer that outputs two logits (AD vs. NC), converted to probabilities via softmax during inference.

### Adaptations for Medical Imaging

While ConvNeXt was originally designed for RGB images, I made several modifications to make it suitable for grayscale MRI:

1. **Single-channel input:** Modified stem to accept 1-channel input.
2. **Dropout regularization:** Added 30% dropout before classification to prevent overfitting on limited medical data.
3. **Grayscale normalization:** Applied channel specific mean/std from ADNI training set.

---

## Replication of Results

To reproduce the reported results, follow the environment setup and execution steps below. All experiments were conducted using PyTorch 2.1, Python 3.9, and an NVIDIA A100 GPU from Rangpur and google colab.

### Dependencies

- python 
- PyTorch 
- TorchVision
- scikit-learn
- Matplotlib
- tqdm
- wandb

```bash
pip install torch torchvision scikit-learn matplotlib tqdm wandb
```

### Running on Rangpur

1. Create a SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=convnext_run
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH -o convnext_%j.out
#SBATCH -e convnext_%j.err

source ~/.bashrc
conda activate torch

pip install --quiet --upgrade tqdm matplotlib scikit-learn wandb timm

wandb login API_KEY

python train.py
```

2. Submit job:

```bash
sbatch script_name
```

### Running on Google Colab

1. Upload all files and adjust paths

2. Place your ADNI dataset inside google drive and update the path at the top of `dataset.py`

   ```python
   ADNI_DATA_PATH = "/content/drive/MyDrive/ADNI/AD_NC"
   ```

3. Enable GPU runtime: Runtime → Change runtime type → A100 GPU

4. Run Training script (`train.py`) and Predict script (`predict.py`)

---

## Data Loading

The ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset was used for this project, containing 2D MRI brain slices from two classes: Alzheimer's Disease (AD) and Cognitively Normal (NC).

### Dataset Structure

```
AD_NC/
 ├── train/
 │   ├── AD/
 │   └── NC/
 ├── test/
 │   ├── AD/
 │   └── NC/
```

Each image file is named with a patient ID prefix such as `388206_78.jpeg`, where `388206` identifies the patient and `_78` refers to the slice index.

### Data Preprocessing and Normalisation

Preprocessing and normalization were handled in `dataset.py`.
All MRI images were:

- Resized to 256×256
- Converted to grayscale (1-channel)
- Normalized using the dataset's computed statistics:
  - Mean: 0.1155
  - Standard deviation: 0.2212

These values were calculated from the training set using `mean_std_calc()` in `utils.py`.
Normalization ensures intensity consistency across scans, which is critical since MRI brightness varies between sessions and scanners.

### Patient Level Data Handling

Initially, the dataset was loaded per image, which led to **data leakage**. Some slices from the same patient appeared in both training and validation sets, increasing accuracy.
To fix this, patient aware splitting was implemented.

Key Improvements included:

1. **Patient ID extraction:**
   Each image filename is parsed to obtain the patient ID (e.g., `388206_78.jpeg` → `388206`).

2. **Patient-based splitting:**
   Using `train_test_split()` on unique patient IDs ensures that all scans from a single patient appear only in one split.

3. **Dataset classes:**
   - `ADNIDataset` – image level dataset (used for fast experiments)
   - `ADNIPatientDataset` – patient grouped dataset supporting:
     - `'single'` mode: randomly select one scan per patient per epoch
     - `'all'` mode: return all scans for patient-level aggregation (used in testing)

This change was crucial to achieve stable validation performance.

### Data Augmentation

To improve generalization and prevent overfitting, several augmentations were applied using `torchvision.transforms`.

#### 1. Training Augmentations

```python
transforms.Resize(IMAGE_SIZE),
transforms.RandomHorizontalFlip(p=0.5),
transforms.RandomRotation(15),
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
transforms.ColorJitter(brightness=0.2, contrast=0.2),
transforms.Grayscale(num_output_channels=1),
transforms.ToTensor(),
transforms.Normalize(mean, std),
transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
```

#### 2. Validation / Test Augmentations

```python
transforms.Resize(IMAGE_SIZE),
transforms.Grayscale(num_output_channels=1),
transforms.ToTensor(),
transforms.Normalize(mean,std)
```

---

## Training

The model was trained to classify MRI slices from the ADNI dataset into Alzheimer's Disease (AD) and Cognitively Normal (NC). Training emphasized reproducibility, stability, and generalization across patient scans.

### Final Parameters

| Parameter         | Value                                                              |
| ----------------- | ------------------------------------------------------------------ |
| Model             | ConvNeXt-Small                                                     |
| Optimizer         | **AdamW**                                                          |
| Learning Rate     | 1e-4                                                               |
| Weight Decay      | 0.05                                                               |
| Batch Size        | 28                                                                 |
| Epochs            | 65                                                                 |
| LR Scheduler      | **ReduceLROnPlateau (mode='max', patience=6)**                     |
| Loss Function     | CrossEntropyLoss (label smoothing=0.1, class weights=[1.65, 1.35]) |
| Regularization    | Dropout = 0.3, Random Erasing = 0.3                                |
| GPU               | NVIDIA A100                                                        |
| Training Duration | ~2 hours                                                           |

Training and validation were logged using **Weights & Biases (W&B)** for real time experiment tracking and performance comparison across different runs.

### Optimisation

Early experiments used various learning rate schedulers (`StepLR`, `CosineAnnealingLR`, `OneCycleLR`), but `ReduceLROnPlateau` provided the most consistent improvements. Mixed precision training using `torch.amp.autocast` was used to stabilize and speed up convergence.

Key training features:

- **Label smoothing (0.1):** Prevents overconfidence and improves calibration.
- **Class weights:** Counteracts dataset imbalance (AD < NC).
- **Model checkpointing:** Saves the best performing model based on validation F1.
- **Full schedule completion:** Early stopping disabled to allow full 65 epoch training, but best path was used from epoch 62.

### Loss Function

A weighted Cross-Entropy Loss was used to handle class imbalance between AD and NC patients. This ensured the model did not bias heavily toward the majority (NC) class.

```python
criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1,
    weight=torch.tensor([1.65, 1.35], dtype=torch.float32).to(DEVICE)
)
```

### Training and Validation Performance

![Training Metrics](README%20resources/training_metrics.png)

- Validation accuracy stabilized around 86%.
- F1-score plateaued near 0.86 by epoch 62.
- Loss curves showed smooth convergence with minimal divergence between train and validation, indicating reduced overfitting.

### Weights and Biases Tracking

The W&B dashboard was used to track multiple training runs and visualize learning stability across configurations.

**Validation F1 Comparison:**
![Validation F1](README%20resources/vallf1.png)

**Validation Accuracy Comparison:**
![Validation Accuracy](README%20resources/valacc.png)

**Training Accuracy Progression:**
![Training Accuracy](README%20resources/trainacc.png)

These results confirm that the final run `convnext_small_run_final` consistently achieved the best validation F1 and accuracy, showing stable convergence compared to earlier unstable runs.

---

## Testing

Testing was performed using both single-scan evaluation and multi-scan (patient-level) aggregation to measure model robustness across multiple MRI slices from the same patient.

Two test loaders were defined in `dataset.py`:

- **Single-Scan Loader:** Evaluates each MRI slice independently.
- **Aggregated Loader:** Combines predictions from all slices of a patient by averaging class probabilities, providing a more reliable patient-level diagnosis.

During inference, the model predicted the likelihood of each class (AD or NC) using a softmax output. All predictions were generated using the best checkpoint (`convnext_small_best.pth`) from epoch 62, selected based on highest validation F1 score (0.868).

---

## Results

### Final Evaluation Metrics

| **Metric** | **Validation** | **Test (Single)** | **Test (Aggregated)** |
| ---------- | -------------- | ----------------- | --------------------- |
| Accuracy   | 86.81%         | 73.56%            | **75.11%**            |
| F1-score   | **0.868**      | 0.729             | **0.743**             |

### Classification Report (Aggregated)

| Class                | Precision | Recall | F1-score | Support |
| -------------------- | --------- | ------ | -------- | ------- |
| AD                   | 0.88      | 0.58   | 0.70     | 223     |
| NC                   | 0.69      | 0.92   | 0.79     | 227     |
| **Overall Accuracy** |           |        | **0.75** | 450     |
| **Macro Avg**        | 0.78      | 0.75   | 0.74     |         |
| **Weighted Avg**     | 0.78      | 0.75   | 0.74     |         |

### Confusion Matrix (Aggregated)

![Confusion Matrix](README%20resources/test_confusion_agg.png)

- The model correctly classifies 92% of NC cases and 58% of AD cases.
- The imbalance reflects the inherent difficulty of distinguishing early stage AD patterns.

### Visualisation of Test Accuracies and F1 Scores

The following comparisons are from Weights and Biases:

**Single Scan Accuracy:**
![Single Scan Accuracy](README%20resources/test%20acc%20single.png)

**Aggregated Accuracy:**
![Aggregated Accuracy](README%20resources/test%20acc.png)

**Single Scan F1:**
![Single Scan F1](README%20resources/f1single.png)

**Aggregated F1:**
![Aggregated F1](README%20resources/f1ag.png)

### Interpretation

- The NC class achieved higher recall, indicating the model effectively identifies cognitively normal subjects.
- The AD class had lower recall (0.58), highlighting challenges in differentiating mild AD cases.
- Aggregated predictions improved both accuracy and F1 by reducing false negatives on AD.
- Overall, the ConvNeXt-Small architecture captured robust spatial biomarkers of AD while maintaining good generalization across unseen patients.

### Sample Predictions

The following sample predictions were visualised using `predict.py`:

#### Single Image Predictions

- **AD Single Prediction:**
![AD Prediction](README%20resources/single_ad.png)

- **NC Single Prediction:**
![NC Prediction](README%20resources/single_nc.png)

#### Balanced Batch Predictions

Predictions shown in green (correct) and red (incorrect).

![Batch Predictions 1](README%20resources/nc_ad1.png)
![Batch Predictions 2](README%20resources/nc_ad2.png)

---

## Discussion

The ConvNeXt-Small model demonstrated consistent convergence and strong validation performance, achieving 86.8% validation accuracy (F1 = 0.868) and 75.1% aggregated test accuracy (F1 = 0.743). These results confirm that the network effectively learned meaningful structural patterns from brain MRIs, particularly for differentiating cognitively normal controls (NC).

However, class imbalance and subtle anatomical differences in Alzheimer's patients at early stages limited recall for the AD class (0.58). Despite this, multi scan aggregation improved both accuracy and F1 by over 1.5%, showing that averaging slice-level predictions helps mitigate noise and bias.

### Key Observations

- Patient-aware splitting eliminated over optimistic validation metrics caused by leakage and produced more realistic results.
- Weighted loss + label smoothing helped stabilize training and improved minority class performance.
- ReduceLROnPlateau dynamically adapted the learning rate, preventing premature convergence observed with StepLR and OneCycleLR.
- Dropout regularization and gentle augmentations controlled overfitting while preserving fine structural details in MRI slices.
- W&B experiment tracking confirmed that the final configuration achieved the most stable learning curve and generalization trend.

Overall, the ConvNeXt-Small backbone proved well suited for 2D medical imaging tasks, offering a balance between capacity and efficiency. The remaining performance gap toward the 80% test accuracy target can be reached with more data and training improvements.

---

## Future Improvements

To further improve model performance and move closer to the 80% test accuracy target, future work will focus on refining the patient-level weighting and aggregation strategy to better combine multi scan predictions and exploring additional techniques to enhance overall model robustness and generalization.

---

## Conclusion

This project successfully implemented and optimized a ConvNeXt-Small architecture for Alzheimer's disease classification on the ADNI MRI dataset. Through patient level data handling, balanced training, and careful tuning of loss and learning rate schedules, the model achieved 86.8% validation accuracy and 75.1% test accuracy (F1 = 0.743).

---

## References

1. Liu, Z., Mao, H., Wu, C., Feichtenhofer, C., Darrell, T., & Xie, S. (2022).  
   A ConvNet for the 2020s (ConvNeXt).  
   arXiv preprint arXiv:2201.03545.  
   [https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545)

2. Facebook Research. (2022).  
   ConvNeXt Model Implementation (GitHub Repository).  
   [https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py)

3. GeeksforGeeks. (2023).  
   Data Preprocessing in PyTorch – Deep Learning Tutorial.  
   [https://www.geeksforgeeks.org/deep-learning/data-preprocessing-in-pytorch/](https://www.geeksforgeeks.org/deep-learning/data-preprocessing-in-pytorch/)

4. freeCodeCamp. (2022).  
   How to Write Better Git Commit Messages.  
   [https://www.freecodecamp.org/news/how-to-write-better-git-commit-messages/](https://www.freecodecamp.org/news/how-to-write-better-git-commit-messages/)

5. Claude AI. (2024).  
   ConvNeXt Architecture Diagram.  
   Generated using Claude AI by Anthropic.  