import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.efficientnet import EfficientNetBaseline


# -------------------------
# Configuration
# -------------------------

TRAIN_PATH = "data/processed/train"
VAL_PATH = "data/processed/val"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -------------------------
# Transforms
# -------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------
# Dataset
# -------------------------

train_dataset = datasets.ImageFolder(
    TRAIN_PATH,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_PATH,
    transform=val_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -------------------------
# Model
# -------------------------

num_classes = len(train_dataset.classes)

model = EfficientNetBaseline(
    num_classes=num_classes
)

model = model.to(DEVICE)


# -------------------------
# Loss + Optimizer
# -------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -------------------------
# Training
# -------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )


# -------------------------
# Save model
# -------------------------

torch.save(
    model.state_dict(),
    "efficientnet_baseline.pth"
)

print("Model saved successfully.")