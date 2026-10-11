import torch
import torch.nn as nn

from models.Stage import Stage


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Initial feature extraction for CIFAR-10
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Four residual stages
        self.stage1 = Stage(
            in_channels=64,
            out_channels=64,
            stride=1,
            num_blocks=2
        )

        self.stage2 = Stage(
            in_channels=64,
            out_channels=128,
            stride=2,
            num_blocks=2
        )

        self.stage3 = Stage(
            in_channels=128,
            out_channels=256,
            stride=2,
            num_blocks=2
        )

        self.stage4 = Stage(
            in_channels=256,
            out_channels=512,
            stride=2,
            num_blocks=2
        )

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Residual stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x

# model = ResNet18(num_classes=10)

# # Simulate one CIFAR-10 batch
# x = torch.randn(64, 3, 32, 32)

# # Forward pass
# model.eval()

# with torch.no_grad():
#     output = model(x)

# print("Input shape: ", x.shape)
# print("Output shape:", output.shape)
# print("Parameters:", sum(
#     p.numel() for p in model.parameters()
# ))

# # Check the output dimensions
# assert output.shape == (64, 10)

# # Check backward propagation
# model.train()

# x = torch.randn(4, 3, 32, 32)
# labels = torch.randint(0, 10, (4,))

# criterion = torch.nn.CrossEntropyLoss()
# output = model(x)
# loss = criterion(output, labels)

# loss.backward()

# print("Loss:", loss.item())
# print("Backward pass successful")