import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.CNN_v3 import CNN


# ---- 1. Data ----

transform = transforms.Compose([
    transforms.ToTensor(),
])

full_train_ds = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_ds = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
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

model = CNN().to(device)


# ---- 3. Loss & optimizer ----

criterion = nn.CrossEntropyLoss()

#optimizer = optim.SGD(model.parameters(),lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ---- 4. Training loop ----

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
        f"Time: {epoch_time:.2f}s"
    )

total_time = time.perf_counter() - start_time
print(f"Total training time: {total_time:.2f}s")

# ---- 5. Final Test Evaluation ----

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