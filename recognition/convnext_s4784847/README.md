# Alzheimer's Disease Classification using ConvNeXt-Small
Name :- Wasana Nawamini Udabage 
StudentNo :- 47848476

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

## ConvNeXt Overview
ConvNeXt is a modern interpretation of convolutional newural networks(ConvNets), often refered to as a "ConvNet for the 2020s". It was introduced by Liu et al.(2022) as a way to bring traditional ConvNets up to the same level of performance as Vision Transformers, while preserving the efficieny and simplicity of convolutional models.

The model combines design concepts from ResNet and Vision Transformers, taking the best from both. From ResNet, ConvNeXt keeps he general stage based architecture and residual connections. From VIT, it adopts concepts such as patch embedding, large receptive fields, layer normalisationa dn GELU activations. 

The authors of the paper 
-- maybe explain a bit about what they did in the begining 

The major architectureal updates for ConvNext were:
- Replacing the initial RedNet stem with a 4*4 convolution(strid 4), similar to how ViTs create pathc tokens.
- Using 7*7 depthwise convolutions inside each blck to capture a lrdger spatial context
- Expanding and projecting channels using an inverted bottlenexk structure
- Switching from BatchNorm + ReLU to LayerNorm + GELU for smoother and more stable leanring 
- Simplifying the overall layout into four clear stafes wirh dowunsampling between each one.

# Why is it suitable for Medical Imaging?

I selected ConvNeXt since its design aligns well with the challenges of medical image analysis, particularly for MRI-based Alzheimer’s disease classification. The model’s large 7×7 depthwise convolutions enable it to capture broad spatial relationships across brain regions, which is crucial for detecting subtle structural changes associated with Alzheimer’s. Its use of Layer Normalisation and GELU activation helps maintain stable training on grayscale MRI data, which typically have lower contrast and variability than natural images. The hierarchical stage structure allows ConvNeXt to extract both local texture details and global anatomical patterns, improving classification robustness. Overall, ConvNeXt combines the simplicity and efficiency of convolutional networks with transformer inspired design principles, offering strong performance and interpretability while remaining computationally efficient for medical imaging tasks.

## Problem Space


## Project Structure 

The implementation is structured into four main files for clarity and maintainability:

- **dataset.py:** Handles ADNI data loading, augmentation, and patient-level splitting.
- **modules.py:** Defines the ConvNeXt-Small model with ConvNeXtBlock and DownsampleLayer.
- **train.py:** Contains the training, validation, and testing loops with W&B logging.
- **utils.py:** Provides data normalization utilities (mean and standard deviation computation).

## Model Architecture 


## Data loading 

### Data preprocessing and Normalisation 
### Pateient level data handling 


## Training the model 

## Testing 


## Results 
# Performance Evaluation Metrics


## Discussion 


## Improvements 

## Conclusion

## How to run the system 

used a100 
initially atated with testingi n ranptr and moved to google colab afterwards 

mention the file path 
slurm script 
how I used wights and biases 

## Dependancies 


## References





