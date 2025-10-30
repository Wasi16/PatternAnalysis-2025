# Alzheimer's Disease Classification using ConvNeXt-Small

**Name:** Wasana Nawamini Udabage  
**Student No:** 47848476

## Overview

This repository implements a ConvNeXt-Small model for classifying Alzheimer's Disease (AD) vs. Cognitively Normal (NC) patients using MRI scans from the ADNI dataset. Built for COMP3710 Pattern Recognition project 8, this implementation features patient-level data splitting to prevent leakage, robust data augmentation, class balancing, and multi-scan prediction aggregation(i removed multiscan i think ). The project achieves [accuracy]% test accuracy, meeting the required 0.8 threshold, with full experiment tracking via Weights & Biases.

## Table of Contents

ConvNeXt
Chosen Problem
Model
Loading data
Training
loss function
optimiser
Testing
Result
Discussion
Conclution
References
Dependencies
Improvements

---

## ConvNeXt Overview

ConvNeXt is a modern interpretation of convolutional newural networks(ConvNets), often refered to as a "ConvNet for the 2020s". It was introduced by Liu et al.(2022) as a way to bring traditional ConvNets up to the same level of performance as Vision Transformers, while preserving the efficieny and simplicity of convolutional models.

The model combines design concepts from ResNet and Vision Transformers, taking the best from both. From ResNet, ConvNeXt keeps he general stage based architecture and residual connections. From Vission Transformers, it adopts concepts such as patchify stem, large receptive fields, layer normalisationa dn GELU activations.

The authors systematically modernized the standard ResNet architecture through a series of design decisions, each backed by empirical evaluation. They began with a ResNet-50 baseline and progressively incorporated transformer inspired modifications, including adjusting the stage compute ratio, using depthwise convolutions, increasing kernel sizes to 7×7, replacing ReLU with GELU, and substituting Batch Normalization with Layer Normalization. (shorten this bit )

The major architectureal updates for ConvNext include:

- **Patchify Stem:** Replacing the traditional ResNet stem with a 4×4 convolution (stride 4), similar to how Vision Transformers create patch tokens
- **Large Depthwise Convolutions:** Using 7×7 depthwise convolutions inside each block to capture a larger spatial context
- **Inverted Bottleneck Structure:** Expanding and projecting channels using an inverted bottleneck (narrow → wide → narrow)
- **Modern Normalization and Activation:** Switching from BatchNorm + ReLU to LayerNorm + GELU for smoother and more stable learning
- **Simplified Design:** Four clear stages with downsampling between each one, creating a hierarchical feature representation
  .

### Why is it suitable for Medical Imaging?

I selected ConvNeXt-Small for this task because its design aligns well with the challenges of medical image analysis, particularly for MRI-based Alzheimer's disease classification:

1. **Large Receptive Fields:** The 7×7 depthwise convolutions enable the model to capture broad spatial relationships across brain regions, which is crucial for detecting subtle structural changes associated with Alzheimer's disease such as hippocampal atrophy and ventricular enlargement.

2. **Stable Training on Grayscale Data:** The use of Layer Normalization and GELU activation helps maintain stable training on grayscale MRI data, which typically have lower contrast and higher variability than natural RGB images.

3. **Hierarchical Feature Extraction:** The four-stage architecture allows ConvNeXt to extract both local texture details (early stages) and global anatomical patterns (later stages), improving classification robustness across different brain regions.

4. **Computational Efficiency:** ConvNeXt combines the simplicity and efficiency of convolutional networks with transformer-inspired design principles, offering strong performance while remaining computationally tractable for medical imaging tasks.

---

## Problem Space

The focus of this project is on the classification of Alzheimer’s disease using MRI scans from the ADNI (Alzheimer’s Disease Neuroimaging Initiative) dataset. Alzheimer’s causes gradual structural changes in the brain, and early detection is crucial for treatment and diagnosis. The goal is to build a model capable of distinguishing between Alzheimer’s patients (AD) and cognitively normal controls (NC) based on 2D MRI slices. This task represents a real world medical image classification challenge, where accurate pattern recognition from subtle anatomical differences is essential.

---

## Project Structure

The implementation is structured into four main files for clarity and maintainability:

- **dataset.py:** Handles ADNI data loading, augmentation, and patient-level splitting.
- **modules.py:** Defines the ConvNeXt-Small model with ConvNeXtBlock and DownsampleLayer.
- **train.py:** Contains the training, validation, and testing loops with W&B logging.
- **utils.py:** Provides data normalization utilities (mean and standard deviation computation).
- **predict.py:** .

---

## Model Architecture

The model used in this project is **ConvNeXt-Small**, a modernized convolutional neural network designed to combine the strengths of classic ConvNets like ResNet with concepts from Vision Transformers.

ConvNeXt-Small consists of four hierarchical stages with progressively increasing channel depth and decreasing spatial resolution:

**Stage Configuration:**

- **Stage 1:** 3 ConvNeXt blocks, 96 channels
- **Stage 2:** 3 ConvNeXt blocks, 192 channels
- **Stage 3:** 27 ConvNeXt blocks, 384 channels
- **Stage 4:** 3 ConvNeXt blocks, 768 channels

![ConvNeXt Block](recognition\convnext_s4784847\README resources\image.png)

**Total Parameters:** ~50M parameters (trainable)

### Key Components

#### 1. Patchify Stem

```python
nn.Conv2d(in_channels=1, out_channels=96, kernel_size=4, stride=4)
nn.LayerNorm(96)
```

Converts input (256×256) into non-overlapping 4×4 patches, reducing spatial dimensions to 64×64.

#### 2. ConvNeXt Block

Each block contains:

- **Depthwise Convolution (7×7):** Captures large spatial context
- **Layer Normalization:** Stabilizes training
- **Pointwise Expansion (1×1):** Expands channels by 4×
- **GELU Activation:** Smooth non-linearity
- **Pointwise Projection (1×1):** Projects back to original dimensions
- **Residual Connection:** Enables gradient flow

#### 3. Downsampling Layer

Between stages, a 2×2 convolution with stride 2 halves spatial dimensions while doubling channels.

#### 4. Classification Head

- Global Average Pooling over spatial dimensions
- Layer Normalization
- Linear layer mapping 768 → 2 classes (AD, NC)

### Adaptations for Medical Imaging

While ConvNeXt was originally designed for RGB images, I made several modifications to make it suitable for grayscale MRI:

1. **Single-channel input:** Modified stem to accept 1-channel input (vs. 3-channel RGB)
2. **Dropout regularization:** Added 30% dropout before classification to prevent overfitting on limited medical data
3. **Grayscale normalization:** Applied channel-specific mean/std from ADNI training set
4. **Large receptive fields:** 7×7 convolutions particularly beneficial for capturing distributed brain atrophy patterns
---

## Data loading

The dataset is saved in the files in the following format
.
├── AD_NC  
 ├── Test  
 ├── AD  
 ├── NC  
 ├── Train
├── AD  
 ├── NC

### Data preprocessing and Normalisation

### Pateient level data handling

### Data Augmentation

## Training the model

## Testing

## Results

# Performance Evaluation Metrics

## Discussion

## Improvements

## Conclusion

## How to run the system

` pip install torch torchvision scikit-learn matplotlib tqdm wandb`

**HPC (Rangpur) Training**

Create a SLURM script (`run_train.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=convnext_adni
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

module load cuda/11.8
module load python/3.9

source /path/to/venv/bin/activate

python train.py
```

Submit job:

```bash
sbatch run_train.sh
```

**Google Colab**

Upload all files and adjust paths:

```python
ADNI_DATA_PATH = "/content/drive/MyDrive/ADNI/AD_NC"
```

Enable GPU runtime: Runtime → Change runtime type → A100 GPU

## Dependancies

python
PyTorch
TorchVision
scikit-learn
Matplotlib
tqdm
wandb

## References
