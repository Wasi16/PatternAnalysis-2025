"""
modules.py
---------

required layers , Convext blovk. downsampler , then convnect tieny

"""
import torch
import torch.nn as nn 
import torch.nn.functional as F

class ConvNeXtBlock(nn.Module):

    def __init__(self, dimentions):
        super().__init__()

        # Depthwise Convolution
        self.dwconv = nn.Conv2d(dimentions, dimentions, kernel_size=7, padding=3, groups=dimentions)

        # Layer Normalisation
        self.norm = nn.LayerNorm(dimentions,eps=1e-6)

        # First Pointwise Convolution(Expansion)
        self.pwconv1 = nn.Linear(dimentions,  4*dimentions)
        # Activation Function
        self.act = nn.GELU
        # Second Pointwise Convolution(Projection)
        self.pwconv2 = nn.Linear(4* dimentions, dimentions)

    def forward(self,x):

        residual = x

        x = self.dwconv(x)

        x = x.permute(0,2,3,1)
        x = self.norm(x)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = x.permute(0,3,1,2)

        x = residual + x

        return x

class DownsampleLayer():

    def __init__(self):
        pass

class ConvNeXt_T():

    def __init__(self):
        pass