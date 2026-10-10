import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import os

if os.path.exists("/content/drive/MyDrive/CIFAR-10-data"):
    data_root = "/content/drive/MyDrive/CIFAR-10-data"
else:
    data_root = "./data"


# ---- 1. Data ----

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])

full_train_ds = datasets.CIFAR10(
    root=data_root,
    train=True,
    download=False,
    transform=transform
)

test_ds = datasets.CIFAR10(
    root=data_root,
    train=False,
    download=False,
    transform=transform
)

# Split the 50,000 training images
train_ds, valid_ds = torch.utils.data.random_split(
    full_train_ds,
    [45000, 5000]
)

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True
)

valid_loader = DataLoader(
    valid_ds,
    batch_size=1000,
    shuffle=False
)

test_loader = DataLoader(
    test_ds,
    batch_size=1000,
    shuffle=False
)