import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)

        # Residual branch: 64 -> 128 -> 128
        self.conv2 = nn.Conv2d(
            64, 128,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(
            128, 128,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)

        # Shortcut projection: 64 -> 128 channels
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                64, 128,
                kernel_size=1
            ),
            nn.BatchNorm2d(128)
        )

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.linear = nn.Linear(
            128 * 8 * 8,
            num_classes
        )

    def forward(self, x):
        # Initial feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Save input for the shortcut
        identity = self.shortcut(x)

        # Residual branch
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Residual addition
        out = out + identity
        out = self.relu(out)

        # Classification head
        out = self.maxpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.linear(out)

        return out