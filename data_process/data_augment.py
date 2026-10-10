import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_loaders(
    batch_size=64,
    eval_batch_size=1000,
    seed=42,
    data_root=None,
):
    if data_root is None:
        if os.path.exists("/content/drive/MyDrive/CIFAR-10-data"):
            data_root = "/content/drive/MyDrive/CIFAR-10-data"
        else:
            data_root = "./data"

    # ---- 1. Transforms ----

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        ),
    ])

    # ---- 2. Separate dataset instances ----

    train_aug_ds = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=train_transform
    )

    train_eval_ds = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=eval_transform
    )

    test_ds = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=False,
        transform=eval_transform
    )

    # ---- 3. Reproducible train/validation split ----

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(train_aug_ds),
        generator=generator
    ).tolist()

    train_indices = indices[:45000]
    valid_indices = indices[45000:]

    train_ds = Subset(train_aug_ds, train_indices)
    valid_ds = Subset(train_eval_ds, valid_indices)

    # ---- 4. DataLoaders ----

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=eval_batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False
    )

    return train_loader, valid_loader, test_loader