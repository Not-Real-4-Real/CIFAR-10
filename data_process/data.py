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
    # ---- 1. Data root ----

    if data_root is None:
        if os.path.exists("/content/drive/MyDrive/CIFAR-10-data"):
            data_root = "/content/drive/MyDrive/CIFAR-10-data"
        else:
            data_root = "./data"

    # ---- 2. Transforms: no augmentation ----

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    # ---- 3. Separate dataset instances ----
    # Both training datasets use the same transform.
    # Separate instances allow us to use different loaders
    # for training and evaluation.

    train_ds_full = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=transform,
    )

    train_eval_ds_full = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=transform,
    )

    test_ds = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=False,
        transform=transform,
    )

    # ---- 4. Reproducible train/validation split ----

    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        len(train_ds_full),
        generator=generator,
    ).tolist()

    train_indices = indices[:45000]
    valid_indices = indices[45000:]

    train_ds = Subset(train_ds_full, train_indices)
    valid_ds = Subset(train_eval_ds_full, valid_indices)
    train_eval_ds = Subset(train_eval_ds_full, train_indices)

    # ---- 5. DataLoaders ----

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=eval_batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
    )

    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=eval_batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        valid_loader,
        test_loader,
        train_eval_loader,
    )