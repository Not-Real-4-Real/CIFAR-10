import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1
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

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)

        x = self.linear(x)

        return x