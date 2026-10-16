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

        assert self.image_size % patch_size == 0
        assert d_model % num_heads == 0

        # ---- Patch embedding ----

        num_patches_per_side = (
            self.image_size // patch_size
        )

        num_patches = num_patches_per_side ** 2

        patch_dim = (
            self.num_channels
            * patch_size
            * patch_size
        )

        self.patch_embedding = nn.Linear(
            patch_dim,
            d_model
        )

        # ---- CLS token ----

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model) * 0.02
        )

        # ---- Positional embedding ----

        # Learnable positional embeddings.
        # Sequence length = patches + CLS token.

        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                num_patches + 1,
                d_model
            ) * 0.02
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

        # Initialize encoder layers independently.
        # TransformerEncoder initially clones the same initialized layer parameters.

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

        # ---- Convert image to patches ----

        p = self.patch_size

        x = x.unfold(2, p, p)
        x = x.unfold(3, p, p)

        # (B, 3, 8, 8, 4, 4)

        x = x.permute(0, 2, 3, 1, 4, 5)

        # (B, 8, 8, 3, 4, 4)

        x = x.reshape(B, -1, self.num_channels * p * p)

        # (B, 64, 48)

        # ---- Project patches into embedding space ----

        x = self.patch_embedding(x)

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


import torch

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
