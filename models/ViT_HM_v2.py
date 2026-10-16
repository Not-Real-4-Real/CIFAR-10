import torch
import torch.nn as nn


class ViT(nn.Module):

    def __init__(
        self,
        num_classes=10,
        patch_size=4,
        d_model=192,
        num_heads=3,
        d_ff=384,
        num_layers=6,
        dropout=0.0
    ):
        super().__init__()

        # ---- Configuration ----

        self.patch_size = patch_size
        self.image_size = 32
        self.num_channels = 3
        self.d_model = d_model

        assert self.image_size % patch_size == 0
        assert d_model % num_heads == 0

        # ---- Overlapping patch embedding ----
        #
        # Kernel size = 2 * patch_size - 1 = 7
        # Stride = patch_size = 4
        # Padding = kernel_size // 2 = 3
        #
        # Neighboring patches overlap because kernel_size > stride.
        #
        # Input:  (B, 3, 32, 32)
        # Output: (B, 192, 8, 8)

        kernel_size = 2 * patch_size - 1
        stride = patch_size
        padding = kernel_size // 2

        self.patch_embedding = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        num_patches_per_side = self.image_size // patch_size
        num_patches = num_patches_per_side ** 2

        # ---- CLS token ----

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model) * 0.02
        )

        # ---- Positional embedding ----

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, d_model) * 0.02
        )

        self.pos_dropout = nn.Dropout(dropout)

        # ---- Transformer encoder ----

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # ---- Classification head ----

        self.final_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Linear(
            d_model,
            num_classes
        )

        # ---- Initialize encoder layers independently ----

        for block in self.encoder.layers:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x):

        # Input: (B, 3, 32, 32)

        B, C, H, W = x.shape

        assert C == self.num_channels
        assert H == self.image_size
        assert W == self.image_size

        # ---- Extract overlapping patches and embed them ----

        x = self.patch_embedding(x)

        # (B, 192, 8, 8)

        # Move the embedding dimension to the last axis.
        # Each spatial position becomes one token.

        x = x.flatten(2).transpose(1, 2)

        # (B, 64, 192)

        # ---- Add CLS token ----

        cls = self.cls_token.expand(B, -1, -1)

        x = torch.cat([cls, x], dim=1)

        # (B, 65, 192)

        # ---- Add positional information ----

        x = x + self.pos_embedding
        x = self.pos_dropout(x)

        # ---- Transformer encoder ----

        x = self.encoder(x)

        # ---- Classification ----

        x = self.final_norm(x)

        cls_output = x[:, 0, :]

        # (B, 192)

        logits = self.classifier(cls_output)

        # (B, 10)

        return logits


if __name__ == "__main__":

    model = ViT(num_classes=10)

    x = torch.randn(64, 3, 32, 32)

    with torch.no_grad():
        logits = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", logits.shape)

    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Parameters: {num_params:,}")