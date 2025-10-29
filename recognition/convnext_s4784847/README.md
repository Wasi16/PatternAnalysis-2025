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

- write what  convnext t is here 
emalain the differnce or improvement compared to resnet18 and how it incoperates eleents from transformers(ViT ) to build the model.  

## Problem Space


## Project Structure 

The implementation is modularised into four main files for clarity and maintainability:

- **dataset.py:** Handles ADNI data loading, augmentation, and patient-level splitting.
- **modules.py:** Defines the ConvNeXt-Small model with ConvNeXtBlock and DownsampleLayer.
- **train.py:** Contains the training, validation, and testing loops with W&B logging.
- **utils.py:** Provides data normalization utilities (mean and standard deviation computation).

This modular design ensures separation of concerns and easy extension for future experiments.

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





