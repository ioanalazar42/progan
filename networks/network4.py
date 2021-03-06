'''Defines the architectures of the ProGAN Critics and Generators. 

There are multiple architectures corresponding to the different image sizes that are dealt with, i.e. 4x4, 8x8, 16x16, ...
New, untrained layers are added gradually to an already trained network so the trained parameters are not affected.
'''

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _upsample(x):
    '''Changes a [N, Depth, Height, Width] tensor to [N, Depth, 2 * Height, 2 * Width].'''

    return F.interpolate(x, scale_factor=2, mode='nearest', recompute_scale_factor=False)

def _downsample(x):
    '''Changes a [N, Depth, Height, Width] tensor to [N, Depth, 0.5 * Height, 0.5 * Width].'''

    return F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True, recompute_scale_factor=False)

def _append_constant(x, const):
    '''Appends constant as a new column (the constant is duplicated on each row).
       
       Example:
         input of shape [128, 8192] -> output of shape [128, 8193].'''

    const = const.tile(x.shape[0], 1) # Replicate constant into shape [N, 1] to allow concatenation.
    return torch.cat((x, const), dim=1)

def _clip_range(x, min_clip=-1, max_clip=1):
    '''Brings values of x into range [min_clip, max_clip].'''

    return torch.clamp(x, min_clip, max_clip)

def _leaky_relu(x):
    '''Applies leaky relu activation function with slope 0.2 to input x.'''

    return F.leaky_relu(x, negative_slope=0.2)

def _normalization(x, epsilon=1e-8):
    return x / ((x**2).mean(dim=1, keepdim=True).sqrt() + epsilon)


class EqualizedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

        nn.init.normal_(self.weight)

        # Define scale for the weights.
        num_weights = self.in_features
        self.scale = math.sqrt(2 / num_weights)

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.scale)

class EqualizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=False)

        nn.init.normal_(self.weight)

        # Define scale for the weights.
        num_weights = np.prod(self.kernel_size) * self.in_channels
        self.scale = math.sqrt(2 / num_weights)
    
    def forward(self, x):
        return torch.conv2d(
            input=x,
            weight=self.weight * self.scale,
            stride=self.stride,
            padding = self.padding
        )

class Critic128x128(nn.Module):

    def __init__(self):
        super(Critic128x128, self).__init__()

        # Input is 3x128x128, output is 32x64x64.
        self.conv1 = EqualizedConv2d(3, 32, kernel_size=(4, 4), stride=2, padding=1)

        # (
        # Input 3x64x64, output is 32x64x64.
        self.residual_rgb_conv = EqualizedConv2d(3, 32, kernel_size=(1, 1))
        self.residual_influence = 1
        # )

        # Input is 32x64x64, output is 64x32x32.
        self.conv2 = EqualizedConv2d(32, 64, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 64x32x32, output is 128x16x16.
        self.conv3 = EqualizedConv2d(64, 128, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 128x16x16, output is 256x8x8.
        self.conv4 = EqualizedConv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 256x8x8, output is 512x4x4.
        self.conv5 = EqualizedConv2d(256, 512, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 512x4x4, output is 1024x2x2.
        self.conv6 = EqualizedConv2d(512, 1024, kernel_size=(3, 3))

        # Input is 4097, output is 1.
        self.fc = EqualizedLinear(4097, 1)

    def forward(self, x):
        std = x.std()

        x_residual = x

        x = _leaky_relu(self.conv1(x))

        if self.residual_rgb_conv and self.residual_influence > 0:
            x_residual = _downsample(x_residual)
            x_residual = _leaky_relu(self.residual_rgb_conv(x_residual))
            x = (1 - self.residual_influence) * x + self.residual_influence * x_residual
        else:
            self.residual_rgb_conv = None

        x = _leaky_relu(self.conv2(x))
        x = _leaky_relu(self.conv3(x))
        x = _leaky_relu(self.conv4(x))
        x = _leaky_relu(self.conv5(x))
        x = _leaky_relu(self.conv6(x))
        x = _append_constant(x.view(-1, 4096), std) # Nx4096 -> Nx4097.
        x = self.fc(x)

        return x

class Critic64x64(nn.Module):

    def __init__(self):
        super(Critic64x64, self).__init__()

        # Input 3x64x64, output is 32x64x64.
        self.rgb_conv = EqualizedConv2d(3, 32, kernel_size=(1, 1))

        # Input is 32x64x64, output is 64x32x32.
        self.conv2 = EqualizedConv2d(32, 64, kernel_size=(4, 4), stride=2, padding=1)

        # (
        # Input 3x32x32, output is 64x32x32.
        self.residual_rgb_conv = EqualizedConv2d(3, 64, kernel_size=(1, 1))
        self.residual_influence = 1
        # )

        # Input is 64x32x32, output is 128x16x16.
        self.conv3 = EqualizedConv2d(64, 128, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 128x16x16, output is 256x8x8.
        self.conv4 = EqualizedConv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 256x8x8, output is 512x4x4.
        self.conv5 = EqualizedConv2d(256, 512, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 512x4x4, output is 1024x2x2.
        self.conv6 = EqualizedConv2d(512, 1024, kernel_size=(3, 3))

        # Input is 4097, output is 1.
        self.fc = EqualizedLinear(4097, 1)

    def forward(self, x):
        std = x.std()

        x_residual = x

        x = _leaky_relu(self.rgb_conv(x))
        x = _leaky_relu(self.conv2(x))

        if self.residual_rgb_conv and self.residual_influence > 0:
            x_residual = _downsample(x_residual)
            x_residual = _leaky_relu(self.residual_rgb_conv(x_residual))
            x = (1 - self.residual_influence) * x + self.residual_influence * x_residual
        else:
            self.residual_rgb_conv = None

        x = _leaky_relu(self.conv3(x))
        x = _leaky_relu(self.conv4(x))
        x = _leaky_relu(self.conv5(x))
        x = _leaky_relu(self.conv6(x))
        x = _append_constant(x.view(-1, 4096), std) # Nx4096 -> Nx4097.
        x = self.fc(x)

        return x

    def evolve(self, device):
        critic128x128_model = Critic128x128().to(device)
        
        critic128x128_model.residual_rgb_conv = self.rgb_conv
        critic128x128_model.conv2 = self.conv2
        critic128x128_model.conv3 = self.conv3
        critic128x128_model.conv4 = self.conv4
        critic128x128_model.conv5 = self.conv5
        critic128x128_model.conv6 = self.conv6
        critic128x128_model.fc = self.fc

        return critic128x128_model

class Critic32x32(nn.Module):

    def __init__(self):
        super(Critic32x32, self).__init__()

        # Input 3x32x32, output is 64x32x32.
        self.rgb_conv = EqualizedConv2d(3, 64, kernel_size=(1, 1))

        # Input is 64x32x32, output is 128x16x16.
        self.conv3 = EqualizedConv2d(64, 128, kernel_size=(4, 4), stride=2, padding=1)
        
        # (
        # Input 3x16x16, output is 128x16x16.
        self.residual_rgb_conv = EqualizedConv2d(3, 128, kernel_size=(1, 1))
        self.residual_influence = 1
        # )

        # Input is 128x16x16, output is 256x8x8.
        self.conv4 = EqualizedConv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 256x8x8, output is 512x4x4.
        self.conv5 = EqualizedConv2d(256, 512, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 512x4x4, output is 1024x2x2.
        self.conv6 = EqualizedConv2d(512, 1024, kernel_size=(3, 3))

        # Input is 4097, output is 1.
        self.fc = EqualizedLinear(4097, 1)

    def forward(self, x):
        std = x.std()

        x_residual = x

        x = _leaky_relu(self.rgb_conv(x))
        x = _leaky_relu(self.conv3(x))

        if self.residual_rgb_conv and self.residual_influence > 0:
            x_residual = _downsample(x_residual)
            x_residual = _leaky_relu(self.residual_rgb_conv(x_residual))
            x = (1 - self.residual_influence) * x + self.residual_influence * x_residual
        else:
            self.residual_rgb_conv = None

        x = _leaky_relu(self.conv4(x))
        x = _leaky_relu(self.conv5(x))
        x = _leaky_relu(self.conv6(x))
        x = _append_constant(x.view(-1, 4096), std) # Nx4096 -> Nx4097.
        x = self.fc(x)

        return x

    def evolve(self, device):
        critic64x64_model = Critic64x64().to(device)

        critic64x64_model.residual_rgb_conv = self.rgb_conv
        critic64x64_model.conv3 = self.conv3
        critic64x64_model.conv4 = self.conv4
        critic64x64_model.conv5 = self.conv5
        critic64x64_model.conv6 = self.conv6
        critic64x64_model.fc = self.fc

        return critic64x64_model

class Critic16x16(nn.Module):

    def __init__(self):
        super(Critic16x16, self).__init__()

        # Input 3x16x16, output is 128x16x16.
        self.rgb_conv = EqualizedConv2d(3, 128, kernel_size=(1, 1))

        # Input is 128x16x16, output is 256x8x8.
        self.conv4 = EqualizedConv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1)

        # (
        # Input 3x8x8, output is 256x8x8.
        self.residual_rgb_conv = EqualizedConv2d(3, 256, kernel_size=(1, 1))
        self.residual_influence = 1
        # )

        # Input is 256x8x8, output is 512x4x4.
        self.conv5 = EqualizedConv2d(256, 512, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 512x4x4, output is 1024x2x2.
        self.conv6 = EqualizedConv2d(512, 1024, kernel_size=(3, 3))

        # Input is 4097, output is 1.
        self.fc = EqualizedLinear(4097, 1)

    def forward(self, x):
        std = x.std()

        x_residual = x

        x = _leaky_relu(self.rgb_conv(x))
        x = _leaky_relu(self.conv4(x))

        if self.residual_rgb_conv and self.residual_influence > 0:
            x_residual = _downsample(x_residual)
            x_residual = _leaky_relu(self.residual_rgb_conv(x_residual))
            x = (1 - self.residual_influence) * x + self.residual_influence * x_residual
        else:
            self.residual_rgb_conv = None

        x = _leaky_relu(self.conv5(x))
        x = _leaky_relu(self.conv6(x))
        x = _append_constant(x.view(-1, 4096), std) # Nx4096 -> Nx4097.
        x = self.fc(x)

        return x

    def evolve(self, device):
        critic32x32_model = Critic32x32().to(device)

        critic32x32_model.residual_rgb_conv = self.rgb_conv
        critic32x32_model.conv4 = self.conv4
        critic32x32_model.conv5 = self.conv5
        critic32x32_model.conv6 = self.conv6
        critic32x32_model.fc = self.fc

        return critic32x32_model

class Critic8x8(nn.Module):

    def __init__(self):
        super(Critic8x8, self).__init__()

        # Input 3x8x8, output is 256x8x8.
        self.rgb_conv = EqualizedConv2d(3, 256, kernel_size=(1, 1))

        # Input is 256x8x8, output is 512x4x4.
        self.conv5 = EqualizedConv2d(256, 512, kernel_size=(4, 4), stride=2, padding=1)

        # (
        # Input 3x4x4, output is 512x4x4.
        self.residual_rgb_conv = EqualizedConv2d(3, 512, kernel_size=(1, 1))
        self.residual_influence = 1
        # )

        # Input is 512x4x4, output is 1024x2x2.
        self.conv6 = EqualizedConv2d(512, 1024, kernel_size=(3, 3))

        # Input is 4097, output is 1.
        self.fc = EqualizedLinear(4097, 1)

    def forward(self, x):
        std = x.std()

        x_residual = x

        x = _leaky_relu(self.rgb_conv(x))
        x = _leaky_relu(self.conv5(x))

        if self.residual_rgb_conv and self.residual_influence > 0:
            x_residual = _downsample(x_residual)
            x_residual = _leaky_relu(self.residual_rgb_conv(x_residual))
            x = (1 - self.residual_influence) * x + self.residual_influence * x_residual
        else:
            self.residual_rgb_conv = None

        x = _leaky_relu(self.conv6(x))
        x = _append_constant(x.view(-1, 4096), std) # Nx4096 -> Nx4097.
        x = self.fc(x)

        return x

    def evolve(self, device):
        critic16x16_model = Critic16x16().to(device)

        critic16x16_model.residual_rgb_conv = self.rgb_conv
        critic16x16_model.conv5 = self.conv5
        critic16x16_model.conv6 = self.conv6
        critic16x16_model.fc = self.fc

        return critic16x16_model

class Critic4x4(nn.Module):

    def __init__(self):
        super(Critic4x4, self).__init__()

        # Input is 3x4x4, output is 512x4x4.
        self.rgb_conv = EqualizedConv2d(3, 512, kernel_size=(1, 1))

        # Input is 512x4x4, output is 1024x2x2.
        self.conv6 = EqualizedConv2d(512, 1024, kernel_size=(3, 3))

        # Input is 4097, output is 1.
        self.fc = EqualizedLinear(4097, 1)

    def forward(self, x):
        std = x.std()

        x = _leaky_relu(self.rgb_conv(x))
        x = _leaky_relu(self.conv6(x))
        x = _append_constant(x.view(-1, 4096), std) # Nx4096 -> Nx4097.
        x = self.fc(x)

        return x

    def evolve(self, device):
        critic8x8_model = Critic8x8().to(device)

        critic8x8_model.residual_rgb_conv = self.rgb_conv
        critic8x8_model.conv6 = self.conv6
        critic8x8_model.fc = self.fc

        return critic8x8_model

class Generator4x4(nn.Module):

    def __init__(self):
        super(Generator4x4, self).__init__()
        # Input is a latent space vector of size 512, output is 1024*2*2.
        self.fc = EqualizedLinear(512, 1024 * 2 * 2)

        # Input is 1024x4x4, output is 512x4x4.
        self.conv1 = EqualizedConv2d(1024, 512, kernel_size=(3, 3), padding=1)

        # Input is 512x4x4, output is 3x4x4.
        self.rgb_conv = EqualizedConv2d(512, 3, kernel_size=(1, 1))

    def forward(self, x):
        x = _leaky_relu(self.fc(x)).view(-1, 1024, 2, 2)
        x = _upsample(x)
        x = _leaky_relu(self.conv1(_normalization(x)))
        x = _clip_range(self.rgb_conv(x))
        
        return x

    def evolve(self, device):
        generator8x8_model = Generator8x8().to(device)

        generator8x8_model.fc = self.fc
        generator8x8_model.conv1 = self.conv1
        generator8x8_model.residual_rgb_conv = self.rgb_conv

        return generator8x8_model

class Generator8x8(nn.Module):

    def __init__(self):
        super(Generator8x8, self).__init__()
        # Input is a latent space vector of size 512, output is 1024*2*2.
        self.fc = EqualizedLinear(512, 1024 * 2 * 2)

        # Input is 1024x4x4, output is 512x4x4.
        self.conv1 = EqualizedConv2d(1024, 512, kernel_size=(3, 3), padding=1)

        # (
        # Input is 512x4x4, output is 3x4x4.
        self.residual_rgb_conv = EqualizedConv2d(512, 3, kernel_size=(1, 1))
        self.residual_influence = 1
        # )

        # Input is 512x8x8, output is 256x8x8.
        self.conv2 = EqualizedConv2d(512, 256, kernel_size=(3, 3), padding=1)

        # Input is 256x8x8, output is 3x8x8,
        self.rgb_conv = EqualizedConv2d(256, 3, kernel_size=(1, 1))

    def forward(self, x):
        x = _leaky_relu(self.fc(x)).view(-1, 1024, 2, 2)

        x = _upsample(x)
        x = _leaky_relu(self.conv1(_normalization(x)))
        
        x_residual = x

        x = _upsample(x)
        x = _leaky_relu(self.conv2(_normalization(x)))

        x = _clip_range(self.rgb_conv(x))

        if self.residual_rgb_conv and self.residual_influence > 0:
            x_residual = _clip_range(self.residual_rgb_conv(x_residual))
            x_residual = _upsample(x_residual)
            x = (1 - self.residual_influence) * x + self.residual_influence * x_residual
        else:
            self.residual_rgb_conv = None

        return x

    def evolve(self, device):
        generator16x16_model = Generator16x16().to(device)

        generator16x16_model.fc = self.fc
        generator16x16_model.conv1 = self.conv1
        generator16x16_model.conv2 = self.conv2
        generator16x16_model.residual_rgb_conv = self.rgb_conv

        return generator16x16_model

class Generator16x16(nn.Module):

    def __init__(self):
        super(Generator16x16, self).__init__()
        # Input is a latent space vector of size 512, output is 1024*2*2.
        self.fc = EqualizedLinear(512, 1024 * 2 * 2)

        # Input is 1024x4x4, output is 512x4x4.
        self.conv1 = EqualizedConv2d(1024, 512, kernel_size=(3, 3), padding=1)

        # Input is 512x8x8, output is 256x8x8.
        self.conv2 = EqualizedConv2d(512, 256, kernel_size=(3, 3), padding=1)

        # (
        # Input is 256x8x8, output is 3x8x8.
        self.residual_rgb_conv = EqualizedConv2d(256, 3, kernel_size=(1, 1))
        self.residual_influence = 1
        # )

        # Input is 256x16x16, output is 128x16x16.
        self.conv3 = EqualizedConv2d(256, 128, kernel_size=(3, 3), padding=1)

        # Input is 128x16x16, output is 3x16x16.
        self.rgb_conv = EqualizedConv2d(128, 3, kernel_size=(1, 1))

    def forward(self, x):
        x = _leaky_relu(self.fc(x)).view(-1, 1024, 2, 2)

        x = _upsample(x)
        x = _leaky_relu(self.conv1(_normalization(x)))

        x = _upsample(x)
        x = _leaky_relu(self.conv2(_normalization(x)))
      
        x_residual = x
                
        x = _upsample(x)
        x = _leaky_relu(self.conv3(_normalization(x)))

        x = _clip_range(self.rgb_conv(x))

        if self.residual_rgb_conv and self.residual_influence > 0:
            x_residual = _clip_range(self.residual_rgb_conv(x_residual))
            x_residual = _upsample(x_residual)
            x = (1 - self.residual_influence) * x + self.residual_influence * x_residual
        else:
            self.residual_rgb_conv = None

        return x

    def evolve(self, device):
        generator32x32_model = Generator32x32().to(device)

        generator32x32_model.fc = self.fc
        generator32x32_model.conv1 = self.conv1
        generator32x32_model.conv2 = self.conv2
        generator32x32_model.conv3 = self.conv3
        generator32x32_model.residual_rgb_conv = self.rgb_conv

        return generator32x32_model

class Generator32x32(nn.Module):

    def __init__(self):
        super(Generator32x32, self).__init__()
        # Input is a latent space vector of size 512, output is 1024*2*2.
        self.fc = EqualizedLinear(512, 1024 * 2 * 2)

        # Input is 1024x4x4, output is 512x4x4.
        self.conv1 = EqualizedConv2d(1024, 512, kernel_size=(3, 3), padding=1)

        # Input is 512x8x8, output is 256x8x8.
        self.conv2 = EqualizedConv2d(512, 256, kernel_size=(3, 3), padding=1)

        # Input is 256x16x16, output is 128x16x16.
        self.conv3 = EqualizedConv2d(256, 128, kernel_size=(3, 3), padding=1)

        # (
        # Input is 128x16x16, output is 3x16x16.
        self.residual_rgb_conv = EqualizedConv2d(128, 3, kernel_size=(1, 1))
        self.residual_influence = 1
        # )

        # Input is 128x32x32, output is 64x32x32.
        self.conv4 = EqualizedConv2d(128, 64, kernel_size=(3, 3), padding=1)

        # Input is 64x32x32, output is 3x32x32.
        self.rgb_conv = EqualizedConv2d(64, 3, kernel_size=(1, 1))

    def forward(self, x):
        x = _leaky_relu(self.fc(x)).view(-1, 1024, 2, 2)

        x = _upsample(x)
        x = _leaky_relu(self.conv1(_normalization(x)))

        x = _upsample(x)
        x = _leaky_relu(self.conv2(_normalization(x)))
                
        x = _upsample(x)
        x = _leaky_relu(self.conv3(_normalization(x)))

        x_residual = x
    
        x = _upsample(x)
        x = _leaky_relu(self.conv4(_normalization(x)))

        x = _clip_range(self.rgb_conv(x))

        if self.residual_rgb_conv and self.residual_influence > 0:
            x_residual = _clip_range(self.residual_rgb_conv(x_residual))
            x_residual = _upsample(x_residual)
            x = (1 - self.residual_influence) * x + self.residual_influence * x_residual
        else:
            self.residual_rgb_conv = None

        return x

    def evolve(self, device):
        generator64x64_model = Generator64x64().to(device)

        generator64x64_model.fc = self.fc
        generator64x64_model.conv1 = self.conv1
        generator64x64_model.conv2 = self.conv2
        generator64x64_model.conv3 = self.conv3
        generator64x64_model.conv4 = self.conv4
        generator64x64_model.residual_rgb_conv = self.rgb_conv

        return generator64x64_model

class Generator64x64(nn.Module):

    def __init__(self):
        super(Generator64x64, self).__init__()
        # Input is a latent space vector of size 512, output is 1024*2*2.
        self.fc = EqualizedLinear(512, 1024 * 2 * 2)

        # Input is 1024x4x4, output is 512x4x4.
        self.conv1 = EqualizedConv2d(1024, 512, kernel_size=(3, 3), padding=1)

        # Input is 512x8x8, output is 256x8x8.
        self.conv2 = EqualizedConv2d(512, 256, kernel_size=(3, 3), padding=1)

        # Input is 256x16x16, output is 128x16x16.
        self.conv3 = EqualizedConv2d(256, 128, kernel_size=(3, 3), padding=1)

        # Input is 128x32x32, output is 64x32x32.
        self.conv4 = EqualizedConv2d(128, 64, kernel_size=(3, 3), padding=1)

        # (
        # Input is 64x32x32, output is 3x32x32.
        self.residual_rgb_conv = EqualizedConv2d(64, 3, kernel_size=(1, 1))
        self.residual_influence = 1
        # )

        # Input is 64x64x64, output is 32x64x64.
        self.conv5 = EqualizedConv2d(64, 32, kernel_size=(3, 3), padding=1)

        # Input is 32x64x64, output is 3x64x64.
        self.rgb_conv = EqualizedConv2d(32, 3, kernel_size=(1, 1))

    def forward(self, x):
        x = _leaky_relu(self.fc(x)).view(-1, 1024, 2, 2)

        x = _upsample(x)
        x = _leaky_relu(self.conv1(_normalization(x)))

        x = _upsample(x)
        x = _leaky_relu(self.conv2(_normalization(x)))

        x = _upsample(x)
        x = _leaky_relu(self.conv3(_normalization(x)))

        x = _upsample(x)
        x = _leaky_relu(self.conv4(_normalization(x)))
        
        x_residual = x

        x = _upsample(x)
        x = _leaky_relu(self.conv5(_normalization(x)))
       
        x = _clip_range(self.rgb_conv(x))

        if self.residual_rgb_conv and self.residual_influence > 0:
            x_residual = _clip_range(self.residual_rgb_conv(x_residual))
            x_residual = _upsample(x_residual)
            x = (1 - self.residual_influence) * x + self.residual_influence * x_residual
        else:
            self.residual_rgb_conv = None

        return x

    def evolve(self, device):
        generator128x128_model = Generator128x128().to(device)

        generator128x128_model.fc = self.fc
        generator128x128_model.conv1 = self.conv1
        generator128x128_model.conv2 = self.conv2
        generator128x128_model.conv3 = self.conv3
        generator128x128_model.conv4 = self.conv4
        generator128x128_model.conv5 = self.conv5
        generator128x128_model.residual_rgb_conv = self.rgb_conv

        return generator128x128_model

class Generator128x128(nn.Module):

    def __init__(self):
        super(Generator128x128, self).__init__()
        # Input is a latent space vector of size 512, output is 1024*2*2.
        self.fc = EqualizedLinear(512, 1024 * 2 * 2)

        # Input is 1024x4x4, output is 512x4x4.
        self.conv1 = EqualizedConv2d(1024, 512, kernel_size=(3, 3), padding=1)

        # Input is 512x8x8, output is 256x8x8.
        self.conv2 = EqualizedConv2d(512, 256, kernel_size=(3, 3), padding=1)

        # Input is 256x16x16, output is 128x16x16.
        self.conv3 = EqualizedConv2d(256, 128, kernel_size=(3, 3), padding=1)

        # Input is 128x32x32, output is 64x32x32.
        self.conv4 = EqualizedConv2d(128, 64, kernel_size=(3, 3), padding=1)

        # Input is 64x64x64, output is 32x64x64.
        self.conv5 = EqualizedConv2d(64, 32, kernel_size=(3, 3), padding=1)

        # (
        # Input is 32x64x64, output is 3x64x64.
        self.residual_rgb_conv = EqualizedConv2d(32, 3, kernel_size=(1, 1))
        self.residual_influence = 1
        # )

        # Input is 32x128x128, output is 3x128x128.
        self.conv6 = EqualizedConv2d(32, 3, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        x = _leaky_relu(self.fc(x)).view(-1, 1024, 2, 2)

        x = _upsample(x)
        x = _leaky_relu(self.conv1(_normalization(x)))

        x = _upsample(x)
        x = _leaky_relu(self.conv2(_normalization(x)))

        x = _upsample(x)
        x = _leaky_relu(self.conv3(_normalization(x)))

        x = _upsample(x)
        x = _leaky_relu(self.conv4(_normalization(x)))

        x = _upsample(x)
        x = _leaky_relu(self.conv5(_normalization(x)))

        x_residual = x

        x = _upsample(x)
        x = _clip_range(self.conv6(_normalization(x)))
        
        if self.residual_rgb_conv and self.residual_influence > 0:
            x_residual = _clip_range(self.residual_rgb_conv(x_residual))
            x_residual = _upsample(x_residual)
            x = (1 - self.residual_influence) * x + self.residual_influence * x_residual
        else:
            self.residual_rgb_conv = None

        return x
