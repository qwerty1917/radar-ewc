import torch.nn as nn


class Dcnn(nn.Module):
    def __init__(self, input_channel, num_classes=7):
        super(Dcnn, self).__init__()

        self.conv_layers = nn.Sequential(
            # Image (Cx64x64)
            nn.Conv2d(in_channels=input_channel, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # State (16x32x32)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # State (32x16x16)
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # State (16x8x8)
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

            # State (num_classes)
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)

        return x
