import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_channel, residual=False):
        super().__init__()
        self.residual = residual

        self.residual_module_1 = nn.Sequential(
            # State (Cx64x64)
            nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(0.2),

            # State (32x32x32)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(0.2),
            # State (32x16x16)
        )

        self.up_channel_module = nn.Sequential(
            # State (32x16x16)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2),
            # State (64x16x16)
        )

        self.residual_module_2 = nn.Sequential(
            # State (64x16x16)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2),

            # State (64x8x8)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2)
            # State (64x4x4)
        )

        self.max_pool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.output = nn.Sequential(
            # State (1024x4x4)
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=4, stride=1, padding=0)
            # Output (1)
        )

    def forward(self, x):
        r1 = self.residual_module_1(x)
        if self.residual:
            r1 = r1 + self.max_pool(x)

        u = self.up_channel_module(r1)

        r2 = self.residual_module_2(u)
        if self.residual:
            r2 = r2 + self.max_pool(u)

        o = self.output(r2)
        return o


class Generator(nn.Module):
    def __init__(self, input_channel, residual=False):
        super().__init__()
        self.residual = residual
        # Input_dim = 100
        # Output_dim = (Cx64x64)
        self.intro_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),
            # State (64x4x4)
        )

        self.residual_module_1 = nn.Sequential(
            # State (64x4x4)
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),

            # State (64x8x8)
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),
            # State (64x16x16)
        )

        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=4)

        self.down_channel = nn.Sequential(
            # State (64x16x16)
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True)
            # State (32x16x16)
        )

        self.residual_module_2 = nn.Sequential(
            # State (32x16x16)
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True),

            # State (32x32x32)
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True),
            # State (32x64x64)
        )

        self.output = nn.Sequential(
            # State (32x64x64)
            nn.Conv2d(in_channels=32, out_channels=input_channel, kernel_size=1, stride=1, padding=0),
            # State (Cx64x64)

            nn.Tanh()
        )

    def forward(self, x):
        i = self.intro_module(x)

        r1 = self.residual_module_1(i)
        if self.residual:
            r1 = r1 + self.up_sample(i)

        d = self.down_channel(r1)

        r2 = self.residual_module_2(d)
        if self.residual:
            r2 = r2 + self.up_sample(d)

        o = self.output(r2)
        return o
