'''Defines the architectures of the ProGAN Critics and Generators. There are multiple architectures corresponding to 
    the different image sizes that are dealt with, i.e. 4x4, 8x8, 16x16, ...
    New, untrained layers are added gradually to an already trained network so the trained parameters are not affected.'''

import torch
import torch.nn as nn
import torch.nn.functional as F

def _upsample(x):
    '''Changes a [N, Depth, Height, Width] tensor to [N, Depth, 2 * Height, 2 * Width].'''

    return F.interpolate(x, scale_factor=2, mode='nearest')

def _downsample(x):
    '''Changes a [N, Depth, Height, Width] tensor to [N, Depth, 0.5 * Height, 0.5 * Width].'''

    return F.interpolate(x, scale_factor=0.5, mode='nearest')

class Critic128x128(nn.Module):

    def __init__(self):
        super(Critic128x128, self).__init__()

        # Input is 3x128x128, output is 32x64x64
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(4, 4), stride=2, padding=1)

        # (
        # Input 3x64x64, output is 32x64x64
        self.residual_rgb_conv = nn.Conv2d(3, 32, kernel_size=(1, 1))
        self.residual_influence = 1
        # )

        # Input is 32x64x64, output is 64x32x32
        self.conv2_layernorm = nn.LayerNorm([32, 64, 64])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 64x32x32, output is 128x16x16
        self.conv3_layernorm = nn.LayerNorm([64, 32, 32])
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 128x16x16, output is 256x8x8
        self.conv4_layernorm = nn.LayerNorm([128, 16, 16])
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 256x8x8, output is 512x4x4
        self.conv5_layernorm = nn.LayerNorm([256, 8, 8])
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 512*4*4, output is 1
        self.fc_layernorm = nn.LayerNorm([512 * 4 * 4])
        self.fc = nn.Linear(512 * 4 * 4, 1)

    def forward(self, x):
        x_residual = x

        x = F.relu(self.conv1(x))

        if self.residual_influence > 0:
            x_residual = _downsample(x_residual)  # 3x128x128 -> 3x64x64
            x_residual = F.relu(self.residual_rgb_conv(x_residual))
        else:
            self.residual_rgb_conv = None

        if self.residual_influence > 0:
            x = (1 - self.residual_influence) * x + self.residual_influence * x_residual

        x = F.relu(self.conv2(self.conv2_layernorm(x)))
        x = F.relu(self.conv3(self.conv3_layernorm(x)))
        x = F.relu(self.conv4(self.conv4_layernorm(x)))
        x = F.relu(self.conv5(self.conv5_layernorm(x)))
        x = self.fc(self.fc_layernorm(x.view(-1, 512 * 4 * 4)))

        return x

class Critic64x64(nn.Module):

    def __init__(self):
        super(Critic64x64, self).__init__()

        # Input 3x64x64, output is 32x64x64
        self.rgb_conv = nn.Conv2d(3, 32, kernel_size=(1, 1))

        # Input is 32x64x64, output is 64x64x64
        self.conv2_layernorm = nn.LayerNorm([32, 64, 64])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2, padding=1)

        # (
        # Input 3x32x32, output is 64x32x32
        self.residual_rgb_conv = nn.Conv2d(3, 64, kernel_size=(1, 1))
        self.residual_influence = 1
        # )

        # Input is 64x32x32, output is 128x16x16
        self.conv3_layernorm = nn.LayerNorm([64, 32, 32])
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 128x16x16, output is 256x8x8
        self.conv4_layernorm = nn.LayerNorm([128, 16, 16])
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 256x8x8, output is 512x4x4
        self.conv5_layernorm = nn.LayerNorm([256, 8, 8])
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 512*4*4, output is 1
        self.fc_layernorm = nn.LayerNorm([512 * 4 * 4])
        self.fc = nn.Linear(512 * 4 * 4, 1)

    def forward(self, x):
        x_residual = x

        x = F.relu(self.rgb_conv(x))
        x = F.relu(self.conv2(self.conv2_layernorm(x)))

        if self.residual_influence > 0:
            x_residual = _downsample(x_residual)  # 3x64x64 -> 3x32x32
            x_residual = F.relu(self.residual_rgb_conv(x_residual))
        else:
            self.residual_rgb_conv = None

        if self.residual_influence > 0:
            x = (1 - self.residual_influence) * x + self.residual_influence * x_residual

        x = F.relu(self.conv3(self.conv3_layernorm(x)))
        x = F.relu(self.conv4(self.conv4_layernorm(x)))
        x = F.relu(self.conv5(self.conv5_layernorm(x)))
        x = self.fc(self.fc_layernorm(x.view(-1, 512 * 4 * 4)))

        return x

    def evolve(self):
        critic128x128_model = Critic128x128()

        critic128x128_model.conv3_layernorm = self.conv3_layernorm
        critic128x128_model.conv3 = self.conv3

        critic128x128_model.conv4_layernorm = self.conv4_layernorm
        critic128x128_model.conv4 = self.conv4

        critic128x128_model.conv5_layernorm = self.conv5_layernorm
        critic128x128_model.conv5 = self.conv5

        critic128x128_model.fc_layernorm = self.fc_layernorm
        critic128x128_model.fc = self.fc

        return critic128x128_model

class Critic32x32(nn.Module):

    def __init__(self):
        super(Critic32x32, self).__init__()

        # Input 3x32x32, output is 64x32x32
        self.rgb_conv = nn.Conv2d(3, 64, kernel_size=(1, 1))

        # Input is 64x32x32, output is 128x32x32
        self.conv3_layernorm = nn.LayerNorm([64, 32, 32])
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=2, padding=1)
        
        # (
        # Input 3x16x16, output is 128x16x16
        self.residual_rgb_conv = nn.Conv2d(3, 128, kernel_size=(1, 1))
        self.residual_influence = 1
        # )

        # Input is 128x16x16, output is 256x8x8
        self.conv4_layernorm = nn.LayerNorm([128, 16, 16])
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 256x8x8, output is 512x4x4
        self.conv5_layernorm = nn.LayerNorm([256, 8, 8])
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 512*4*4, output is 1
        self.fc_layernorm = nn.LayerNorm([512 * 4 * 4])
        self.fc = nn.Linear(512 * 4 * 4, 1)

    def forward(self, x):
        x_residual = x

        x = F.relu(self.rgb_conv(x))
        x = F.relu(self.conv3(self.conv3_layernorm(x)))

        if self.residual_influence > 0:
            x_residual = _downsample(x_residual)  # 3x32x32 -> 3x16x16
            x_residual = F.relu(self.residual_rgb_conv(x_residual))
        else:
            self.residual_rgb_conv = None

        if self.residual_influence > 0:
            x = (1 - self.residual_influence) * x + self.residual_influence * x_residual

        x = F.relu(self.conv4(self.conv4_layernorm(x)))
        x = F.relu(self.conv5(self.conv5_layernorm(x)))
        x = self.fc(self.fc_layernorm(x.view(-1, 512 * 4 * 4)))

        return x

    def evolve(self):
        critic64x64_model = Critic64x64()

        critic64x64_model.conv4_layernorm = self.conv4_layernorm
        critic64x64_model.conv4 = self.conv4

        critic64x64_model.conv5_layernorm = self.conv5_layernorm
        critic64x64_model.conv5 = self.conv5

        critic64x64_model.fc_layernorm = self.fc_layernorm
        critic64x64_model.fc = self.fc

        return critic64x64_model

class Critic16x16(nn.Module):

    def __init__(self):
        super(Critic16x16, self).__init__()

        # Input 3x16x16, output is 128x16x16
        self.rgb_conv = nn.Conv2d(3, 128, kernel_size=(1, 1))

        # Input is 128x16x16, output is 256x8x8
        self.conv4_layernorm = nn.LayerNorm([128, 16, 16])
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1)

        # (
        # Input 3x8x8, output is 256x8x8
        self.residual_rgb_conv = nn.Conv2d(3, 256, kernel_size=(1, 1))
        self.residual_influence = 1
        # )

        # Input is 256x8x8, output is 512x4x4
        self.conv5_layernorm = nn.LayerNorm([256, 8, 8])
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 512*4*4, output is 1
        self.fc_layernorm = nn.LayerNorm([512 * 4 * 4])
        self.fc = nn.Linear(512 * 4 * 4, 1)

    def forward(self, x):
        x_residual = x

        x = F.relu(self.rgb_conv(x))
        x = F.relu(self.conv4(self.conv4_layernorm(x)))

        if self.residual_influence > 0:
            x_residual = _downsample(x_residual)  # 3x16x16 -> 3x8x8
            x_residual = F.relu(self.residual_rgb_conv(x_residual))
        else:
            self.residual_rgb_conv = None

        if self.residual_influence > 0:
            x = (1 - self.residual_influence) * x + self.residual_influence * x_residual

        x = F.relu(self.conv5(self.conv5_layernorm(x)))
        x = self.fc(self.fc_layernorm(x.view(-1, 512 * 4 * 4)))

        return x

    def evolve(self):
        critic32x32_model = Critic32x32()

        critic32x32_model.conv5_layernorm = self.conv5_layernorm
        critic32x32_model.conv5 = self.conv5

        critic32x32_model.fc_layernorm = self.fc_layernorm
        critic32x32_model.fc = self.fc

        return critic32x32_model

class Critic8x8(nn.Module):

    def __init__(self):
        super(Critic8x8, self).__init__()

        # Input 3x8x8, output is 256x8x8
        self.rgb_conv = nn.Conv2d(3, 256, kernel_size=(1, 1))

        # Input is 256x8x8, output is 512x4x4
        self.conv5_layernorm = nn.LayerNorm([256, 8, 8])
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(4, 4), stride=2, padding=1)

        # (
        # Input 3x4x4, output is 512x4x4
        self.residual_rgb_conv = nn.Conv2d(3, 512, kernel_size=(1, 1))
        self.residual_influence = 1
        # )

        # Input is 512*4*4, output is 1
        self.fc_layernorm = nn.LayerNorm([512 * 4 * 4])
        self.fc = nn.Linear(512 * 4 * 4, 1)

    def forward(self, x):
        x_residual = x

        x = F.relu(self.rgb_conv(x))
        x = F.relu(self.conv5(self.conv5_layernorm(x)))

        if self.residual_influence > 0:
            x_residual = _downsample(x_residual)  # 3x8x8 -> 3x4x4
            x_residual = F.relu(self.residual_rgb_conv(x_residual))
        else:
            self.residual_rgb_conv = None

        if self.residual_influence > 0:
            x = (1 - self.residual_influence) * x + self.residual_influence * x_residual

        x = self.fc(self.fc_layernorm(x.view(-1, 512 * 4 * 4)))

        return x

    def evolve(self):
        critic16x16_model = Critic16x16()

        critic16x16_model.conv5_layernorm = self.conv5_layernorm
        critic16x16_model.conv5 = self.conv5

        critic16x16_model.fc_layernorm = self.fc_layernorm
        critic16x16_model.fc = self.fc

        return critic16x16_model


class Critic4x4(nn.Module):

    def __init__(self):
        super(Critic4x4, self).__init__()

        # Input 3x4x4, output is 512x4x4
        self.rgb_conv = nn.Conv2d(3, 512, kernel_size=(1, 1))

        # Input is 512*4*4, output is 1
        self.fc_layernorm = nn.LayerNorm([512 * 4 * 4])
        self.fc = nn.Linear(512 * 4 * 4, 1)

    def forward(self, x):
        x = F.relu(self.rgb_conv(x)).view(-1, 512 * 4 * 4)
        x = self.fc(self.fc_layernorm(x))

        return x

    def evolve(self):
        critic8x8_model = Critic8x8()

        critic8x8_model.fc_layernorm = self.fc_layernorm
        critic8x8_model.fc = self.fc

        return critic8x8_model

class Generator4x4(nn.Module):

    def __init__(self):
        super(Generator4x4, self).__init__()
        # Input is a latent space vector of size 512, output is 512*4*4
        self.fc = nn.Linear(512, 512 * 4 * 4)

        # Input is 512x4x4, output is 3x4x4
        self.rgb_conv = nn.Conv2d(512, 3, kernel_size=(1, 1))

    def forward(self, x):
        x = F.relu(self.fc(x)).view(-1, 512, 4, 4)

        x = self.rgb_conv(x)
        return x

    def evolve(self):
        generator8x8_model = Generator8x8()

        generator8x8_model.fc = self.fc

        return generator8x8_model

class Generator8x8(nn.Module):

    def __init__(self):
        super(Generator8x8, self).__init__()
        # Input is a latent space vector of size 512, output is 512*4*4
        self.fc = nn.Linear(512, 512 * 4 * 4)

        # (
        # Input is 512x4x4, output is 3x4x4
        self.residual_rgb_conv = nn.Conv2d(512, 3, kernel_size=(1, 1))
        self.residual_influence = 1
        # upsample
        # )

        # Input is 512x8x8, output is 256x8x8
        self.conv1_bn = nn.BatchNorm2d(512)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=1, padding=1)

        # Input is 256x8x8, output is 3x8x8
        self.rgb_conv = nn.Conv2d(256, 3, kernel_size=(1, 1))

    def forward(self, x):
        x = F.relu(self.fc(x)).view(-1, 512, 4, 4)

        x_residual = x

        x = _upsample(x)
        x = F.relu(self.conv1(self.conv1_bn(x)))
        x = self.rgb_conv(x)

        if self.residual_influence > 0:
            x_residual = self.residual_rgb_conv(x_residual)
            x_residual = _upsample(x_residual)  # 3x4x4 -> 3x8x8

        if self.residual_influence > 0:
            x = (1 - self.residual_influence) * x + self.residual_influence * x_residual
        else:
            self.residual_rgb_conv = None

        return x

    def evolve(self):
        generator16x16_model = Generator16x16()

        generator16x16_model.fc = self.fc

        return generator16x16_model

class Generator16x16(nn.Module):

    def __init__(self):
        super(Generator16x16, self).__init__()
        # Input is a latent space vector of size 512, output is 512*4*4
        self.fc = nn.Linear(512, 512 * 4 * 4)

        # Input is 512x8x8, output is 256x8x8
        self.conv1_bn = nn.BatchNorm2d(512)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=1, padding=1)

        # (
        # Input is 256x8x8, output is 3x8x8
        self.residual_rgb_conv = nn.Conv2d(256, 3, kernel_size=(1, 1))
        self.residual_influence = 1
        # upsample
        # )

        # Input is 256x16x16, output is 128x16x16
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=1, padding=1)

        # Input is 128x16x16, output is 3x16x16
        self.rgb_conv = nn.Conv2d(128, 3, kernel_size=(1, 1))

    def forward(self, x):
        x = F.relu(self.fc(x)).view(-1, 512, 4, 4)

        x = _upsample(x)
        x = F.relu(self.conv1(self.conv1_bn(x)))

        x_residual = x

        x = _upsample(x)
        x = F.relu(self.conv2(self.conv2_bn(x)))
        
        x = self.rgb_conv(x)

        if self.residual_influence > 0:
            x_residual = self.residual_rgb_conv(x_residual)
            x_residual = _upsample(x_residual)  # 3x8x8 -> 3x16x16

        if self.residual_influence > 0:
            x = (1 - self.residual_influence) * x + self.residual_influence * x_residual
        else:
            self.residual_rgb_conv = None

        return x

    def evolve(self):
        generator32x32_model = Generator32x32()

        generator32x32_model.fc = self.fc

        generator32x32_model.conv1_bn = self.conv1_bn
        generator32x32_model.conv1 = self.conv1

        return generator32x32_model


class Generator32x32(nn.Module):

    def __init__(self):
        super(Generator32x32, self).__init__()
        # Input is a latent space vector of size 512, output is 512*4*4
        self.fc = nn.Linear(512, 512 * 4 * 4)

        # Input is 512x8x8, output is 256x8x8
        self.conv1_bn = nn.BatchNorm2d(512)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=1, padding=1)

        # Input is 256x16x16, output is 128x16x16
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=1, padding=1)

        # (
        # Input is 128x16x16, output is 3x16x16
        self.residual_rgb_conv = nn.Conv2d(128, 3, kernel_size=(1, 1))
        self.residual_influence = 1
        # upsample
        # )

        # Input is 128x32x32, output is 64x32x32
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1)

        # Input is 64x32x32, output is 3x32x32
        self.rgb_conv = nn.Conv2d(64, 3, kernel_size=(1, 1))

    def forward(self, x):
        x = F.relu(self.fc(x)).view(-1, 512, 4, 4)

        x = _upsample(x)
        x = F.relu(self.conv1(self.conv1_bn(x)))

        x = _upsample(x)
        x = F.relu(self.conv2(self.conv2_bn(x)))

        x_residual = x

        x = _upsample(x)
        x = F.relu(self.conv3(self.conv3_bn(x)))
        
        x = self.rgb_conv(x)

        if self.residual_influence > 0:
            x_residual = self.residual_rgb_conv(x_residual)
            x_residual = _upsample(x_residual)  # 3x16x16 -> 3x32x32

        if self.residual_influence > 0:
            x = (1 - self.residual_influence) * x + self.residual_influence * x_residual
        else:
            self.residual_rgb_conv = None

        return x

    def evolve(self):
        generator64x64_model = Generator64x64()

        generator64x64_model.fc = self.fc

        generator64x64_model.conv1_bn = self.conv1_bn
        generator64x64_model.conv1 = self.conv1

        generator64x64_model.conv2_bn = self.conv2_bn
        generator64x64_model.conv2 = self.conv2   

        return generator64x64_model

class Generator64x64(nn.Module):

    def __init__(self):
        super(Generator64x64, self).__init__()
        # Input is a latent space vector of size 512, output is 512*4*4
        self.fc = nn.Linear(512, 512 * 4 * 4)

        # Input is 512x8x8, output is 256x8x8
        self.conv1_bn = nn.BatchNorm2d(512)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=1, padding=1)

        # Input is 256x16x16, output is 128x16x16
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=1, padding=1)

        # Input is 128x32x32, output is 64x32x32
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1)

        # (
        # Input is 64x32x32, output is 3x32x32
        self.residual_rgb_conv = nn.Conv2d(64, 3, kernel_size=(1, 1))
        self.residual_influence = 1
        # upsample
        # )

        # Input is 64x64x64, output is 32x64x64
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1)

        # Input is 32x64x64, output is 3x64x64
        self.rgb_conv = nn.Conv2d(32, 3, kernel_size=(1, 1))

    def forward(self, x):
        x = F.relu(self.fc(x)).view(-1, 512, 4, 4)

        x = _upsample(x)
        x = F.relu(self.conv1(self.conv1_bn(x)))

        x = _upsample(x)
        x = F.relu(self.conv2(self.conv2_bn(x)))

        x = _upsample(x)
        x = F.relu(self.conv3(self.conv3_bn(x)))

        x_residual = x

        x = _upsample(x)
        x = F.relu(self.conv4(self.conv4_bn(x)))
       
        x = self.rgb_conv(x)

        if self.residual_influence > 0:
            x_residual = self.residual_rgb_conv(x_residual)
            x_residual = _upsample(x_residual)  # 3x32x32 -> 3x64x64

        if self.residual_influence > 0:
            x = (1 - self.residual_influence) * x + self.residual_influence * x_residual
        else:
            self.residual_rgb_conv = None

        return x

    def evolve(self):
        generator128x128_model = Generator128x128()

        generator128x128_model.fc = self.fc

        generator128x128_model.conv1_bn = self.conv1_bn
        generator128x128_model.conv1 = self.conv1

        generator128x128_model.conv2_bn = self.conv2_bn
        generator128x128_model.conv2 = self.conv2   

        generator128x128_model.conv3_bn = self.conv3_bn
        generator128x128_model.conv3 = self.conv3   

        return generator128x128_model

class Generator128x128(nn.Module):

    def __init__(self):
        super(Generator128x128, self).__init__()
        # Input is a latent space vector of size 512, output is 512*4*4
        self.fc = nn.Linear(512, 512 * 4 * 4)

        # Input is 512x8x8, output is 256x8x8
        self.conv1_bn = nn.BatchNorm2d(512)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=1, padding=1)

        # Input is 256x16x16, output is 128x16x16
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=1, padding=1)

        # Input is 128x32x32, output is 64x32x32
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1)

        # Input is 64x64x64, output is 32x64x64
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1)

        # (
        # Input is 32x64x64, output is 3x64x64
        self.residual_rgb_conv = nn.Conv2d(32, 3, kernel_size=(1, 1))
        self.residual_influence = 1
        # upsample
        # )

        # Input is 32x128x128, output is 3x128x128
        self.conv5_bn = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.fc(x)).view(-1, 512, 4, 4)

        x = _upsample(x)
        x = F.relu(self.conv1(self.conv1_bn(x)))

        x = _upsample(x)
        x = F.relu(self.conv2(self.conv2_bn(x)))

        x = _upsample(x)
        x = F.relu(self.conv3(self.conv3_bn(x)))

        x = _upsample(x)
        x = F.relu(self.conv4(self.conv4_bn(x)))

        x_residual = x

        x = _upsample(x)
        x = F.relu(self.conv5(self.conv5_bn(x)))
        
        if self.residual_influence > 0:
            x_residual = self.residual_rgb_conv(x_residual)
            x_residual = _upsample(x_residual)  # 3x64x64 -> 3x128x128

        if self.residual_influence > 0:
            x = (1 - self.residual_influence) * x + self.residual_influence * x_residual
        else:
            self.residual_rgb_conv = None

        return x

