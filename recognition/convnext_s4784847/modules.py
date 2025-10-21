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

class DownsampleLayer(nn.Module):
    """
    
    """
    def __init__(self, input, output):
        super().__init__()
        # 2x2 convolution with stride 2 halves H and W
        self.conv = nn.Conv2d(input, output, kernel_size=2, stride=2)
        # LayerNorm after downsampling (normalizes per channel)
        self.norm = nn.LayerNorm(output, eps=1e-6)

    def forward(self, x):

        # Apply conv for spatial downsampling
        x = self.conv(x)
        # Switch to channels-last format for LayerNorm
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Back to channels-first format
        x = x.permute(0, 3, 1, 2)
        return x
    
class ConvNeXt_T():
    """
    ConvNeXt Tiny implementation 
    """
    def __init__(self, in_ch=1, num_classes=2):
        super().__init__()

        # Stage configuration (depths and channel sizes)
        self.depths = [3, 3, 9, 3]          # Number of ConvNeXt blocks per stage
        self.dims = [96, 192, 384, 768]     # Channels per stage

        # Stem / Patchify layer
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, self.dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(self.dims[0], eps=1e-6)
        )

        # Build hierarchical stages
        self.stages = nn.ModuleList()
        in_channels = self.dims[0]

        for i in range(len(self.depths)):
            out_channels = self.dims[i]
            num_blocks = self.depths[i]

            # Create N ConvNeXt blocks for this stage
            blocks = [ConvNeXtBlock(dim=out_channels) for _ in range(num_blocks)]

            stage = nn.Sequential(*blocks)
            self.stages.append(stage)

            # Add Downsample layer between stages (except after the last)
            if i < len(self.depths) - 1:
                self.stages.append(DownsampleLayer(out_channels, self.dims[i + 1]))

        # Classification Head
        self.norm = nn.LayerNorm(self.dims[-1], eps=1e-6)
        self.head = nn.Linear(self.dims[-1], num_classes)

        # Weight Initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Applies truncated normal initialization to weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # Stem: patchify input (reduce resolution 4x)
        x = self.stem[0](x)
        x = x.permute(0, 2, 3, 1)
        x = self.stem[1](x)
        x = x.permute(0, 3, 1, 2)

        # Sequentially pass through stages
        for stage in self.stages:
            x = stage(x)

        # Global average pooling
        x = x.mean([-2, -1])  # Average over H and W

        # Final normalization and classification head
        x = self.norm(x)
        x = self.head(x)
        
        return x