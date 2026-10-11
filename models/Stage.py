import torch
import torch.nn as nn

from models.BasicBlock import BasicBlock


class Stage(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        num_blocks=2
    ):
        super().__init__()

        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")

        blocks = []

        # First block may change dimensions
        blocks.append(
            BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride
            )
        )

        # Remaining blocks preserve dimensions
        for _ in range(1, num_blocks):
            blocks.append(
                BasicBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

# # Test 1: Stage 1
# # 64 channels, no downsampling
# stage1 = Stage(
#     in_channels=64,
#     out_channels=64,
#     stride=1,
#     num_blocks=2
# )

# x1 = torch.randn(4, 64, 32, 32)
# y1 = stage1(x1)

# print("Stage 1")
# print("Input: ", x1.shape)
# print("Output:", y1.shape)


# # Test 2: Stage 2
# # Increase channels and downsample
# stage2 = Stage(
#     in_channels=64,
#     out_channels=128,
#     stride=2,
#     num_blocks=2
# )

# x2 = torch.randn(4, 64, 32, 32)
# y2 = stage2(x2)

# print("\nStage 2")
# print("Input: ", x2.shape)
# print("Output:", y2.shape)


# # Verify backward propagation
# loss = y2.mean()
# loss.backward()

# print("\nBackward pass successful")