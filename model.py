import torch.nn as nn


class Dcnn(nn.Module):
    def __init__(self, input_channel):
        super(Dcnn, self).__init__()

        self.conv_layers = nn.Sequential(
            # Image (Cx128x128)
            nn.Conv2d(in_channels=input_channel, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # State (16x64x64)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # State (32x32x32)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # State (64x16x16)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # State (128x8x8)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # State (256x4x4)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # State (512x2x2)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            # State (1024x1x1)
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.4),

            # State (128)
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),

            # State (128)
            nn.Linear(128, 7),
        )

        # self.output = nn.Sequential(
        #     # State (7)
        #     nn.Softmax()
        # )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        # x = self.output(x)

        return x
