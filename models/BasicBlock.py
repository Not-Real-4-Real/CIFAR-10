import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut: identity by default
        self.shortcut = nn.Identity()

        # Project the shortcut when dimensions change
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Preserve the original input
        identity = self.shortcut(x)

        # Main branch
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual addition
        out = out + identity

        # Final activation
        out = self.relu(out)

        return out


# Case 1: Same channels and spatial dimensions
block1 = BasicBlock(
    in_channels=64,
    out_channels=64,
    stride=1
)

x1 = torch.randn(4, 64, 32, 32)
y1 = block1(x1)

print("Case 1")
print("Input: ", x1.shape)
print("Output:", y1.shape)


# Case 2: Increase channels and downsample
block2 = BasicBlock(
    in_channels=64,
    out_channels=128,
    stride=2
)

x2 = torch.randn(4, 64, 32, 32)
y2 = block2(x2)

print("\nCase 2")
print("Input: ", x2.shape)
print("Output:", y2.shape)


# Verify backward propagation
loss = y2.mean()
loss.backward()

print("\nBackward pass successful")
