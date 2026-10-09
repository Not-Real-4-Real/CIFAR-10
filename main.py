import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.CNN_v5 import CNN

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


# ---- 2. Model ----

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model = CNN().to(device)


# ---- 3. Loss & optimizer ----

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.1
)


# ---- 4. Best model tracking ----

best_valid_loss = float("inf")
best_epoch = 0

best_model_path = "cnn_v3_stepLR_best.pth"


# ---- 5. Training loop ----

epochs = 10

start_time = time.perf_counter()

for epoch in range(epochs):

    epoch_start = time.perf_counter()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        preds = outputs.argmax(dim=1)

        train_correct += (
            preds == labels
        ).sum().item()

        train_total += labels.size(0)

    avg_train_loss = (
        train_loss / len(train_loader)
    )

    train_accuracy = (
        100 * train_correct / train_total
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    valid_loss = 0.0
    valid_correct = 0
    valid_total = 0

    with torch.no_grad():

        for images, labels in valid_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            valid_loss += loss.item()

            preds = outputs.argmax(dim=1)

            valid_correct += (
                preds == labels
            ).sum().item()

            valid_total += labels.size(0)

    avg_valid_loss = (
        valid_loss / len(valid_loader)
    )

    valid_accuracy = (
        100 * valid_correct / valid_total
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if avg_valid_loss < best_valid_loss:

        best_valid_loss = avg_valid_loss
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            f"  -> New best model saved "
            f"(Valid Loss: {best_valid_loss:.4f})"
        )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]

    epoch_time = time.perf_counter() - epoch_start

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Valid Loss: {avg_valid_loss:.4f} | "
        f"Valid Acc: {valid_accuracy:.2f}% | "
        f"LR: {current_lr:.6f} | "
        f"Time: {epoch_time:.2f}s"
    )


total_time = time.perf_counter() - start_time

print(f"Total training time: {total_time:.2f}s")
print(
    f"Best model: Epoch {best_epoch} "
    f"| Valid Loss: {best_valid_loss:.4f}"
)


# ---- 6. Load best model ----

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=device
    )
)

print(f"Loaded best model from epoch {best_epoch}")


# ---- 7. Final Test Evaluation ----

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        preds = outputs.argmax(dim=1)

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

test_accuracy = 100 * correct / total

print(
    f"Test accuracy: {test_accuracy:.2f}%"
)