import torch.nn as nn
from torchvision.models import resnet18


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Load the torchvision ResNet18 architecture from scratch
        self.model = resnet18(weights=None)

        # Adapt the initial layers for CIFAR-10 (32x32 images)
        self.model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.model.maxpool = nn.Identity()

        # Replace the ImageNet classifier with a CIFAR-10 classifier (10 classes)
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)

# import torch

# model = ResNet18(num_classes=10)

# x = torch.randn(64, 3, 32, 32)
# y = model(x)

# print(y.shape)
# print(sum(p.numel() for p in model.parameters()))

# assert y.shape == (64, 10)