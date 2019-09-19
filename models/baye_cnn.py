import torch.nn as nn
import torch.nn.functional as F
from models.bayes_layer import BayesianConv2D, BayesianLinear


class Dcnn(nn.Module):
    def __init__(self, input_channel, ratio, multi=False, num_tasks=12):
        super(Dcnn, self).__init__()

        # Image (Cx64x64)
        self.conv1 = BayesianConv2D(in_channels=input_channel, out_channels=16,
                                    kernel_size=7, stride=1, padding=3, ratio=ratio)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.maxpool1 = nn.MaxPool2d(2)

        # State (16x32x32)
        self.conv2 = BayesianConv2D(in_channels=16, out_channels=32,
                                    kernel_size=5, stride=1, padding=2, ratio=ratio)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.maxpool2 = nn.MaxPool2d(2)

        # State (32x16x16)

        self.conv3 = BayesianConv2D(in_channels=32, out_channels=16,
                                    kernel_size=3, stride=1, padding=1, ratio=ratio)
        self.bn3 = nn.BatchNorm2d(num_features=16)
        self.maxpool3 = nn.MaxPool2d(2)

        # State (16x8x8)

        # State (1024x1x1)
        self.fc1 = BayesianLinear(1024, 128, ratio=ratio)
        self.drop1 = nn.Dropout(0.4)
        # State (128)
        self.fc2 = BayesianLinear(128, 128, ratio=ratio)
        self.drop2 = nn.Dropout(0.4)

        # State (128)
        self.multi = multi
        if self.multi:
            self.num_tasks = num_tasks
            self.last = nn.ModuleList()
            for _ in range(self.num_tasks):
                self.last.append(nn.Linear(128, 7))
        else:
            self.last = nn.Linear(128, 7)

    def forward(self, x, head_idx=0, sample=False):
        x = self.maxpool1(F.relu(self.bn1(self.conv1(x, sample))))
        x = self.maxpool2(F.relu(self.bn2(self.conv2(x, sample))))
        x = self.maxpool3(F.relu(self.bn3(self.conv3(x, sample))))

        x = x.reshape(x.size(0), -1)

        x = self.drop1(F.relu(self.fc1(x,sample)))
        x = self.drop2(F.relu(self.fc2(x,sample)))

        if self.multi:
            x = self.last[head_idx](x)
        else:
            x = self.last(x)

        return x

"""
class Dcnn(nn.Module):
    def __init__(self, input_channel, ratio, multi=False, num_tasks=12):
        super(Dcnn, self).__init__()

        self.conv_layers = nn.Sequential(
            # Image (Cx64x64)
            BayesianConv2D(in_channels=input_channel, out_channels=16, kernel_size=7, stride=1, padding=3, ratio=ratio),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # State (16x32x32)
            BayesianConv2D(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2, ratio=ratio),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # State (32x16x16)
            BayesianConv2D(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, ratio=ratio),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # State (16x8x8)
        )

        self.fc_layers = nn.Sequential(
            # State (1024x1x1)
            BayesianLinear(1024, 128, ratio=ratio),
            nn.ReLU(),
            nn.Dropout(0.4),

            # State (128)
            BayesianLinear(128, 128, ratio=ratio),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        # State (128)
        self.multi = multi
        if self.multi:
            self.num_tasks = num_tasks
            self.last = nn.ModuleList()
            for _ in range(self.num_tasks):
                self.last.append(nn.Linear(128, 7))
        else:
            self.last = nn.Linear(128, 7)

    def forward(self, x, head_idx=0):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        if self.multi:
            x = self.last[head_idx](x)
        else:
            x = self.last(x)

        return x
"""