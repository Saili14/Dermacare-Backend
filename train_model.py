import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# ----------------------------
# SETTINGS
# ----------------------------
DATA_DIR = "models/Final_version/dataset"
BATCH_SIZE = 16
EPOCHS = 5   # keep small (fast training)
NUM_CLASSES = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# TRANSFORMS
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# ----------------------------
# DATA
# ----------------------------
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print("Classes:", train_dataset.classes)

# ----------------------------
# MODEL (EfficientNet)
# ----------------------------
model = models.efficientnet_b0(pretrained=True)

# Freeze base layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

model = model.to(DEVICE)

# ----------------------------
# LOSS + OPTIMIZER
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# ----------------------------
# TRAIN LOOP
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}")

# ----------------------------
# SAVE MODEL
# ----------------------------
os.makedirs("models/new_model", exist_ok=True)

torch.save(model.state_dict(), "models/new_model/model.pth")

print("✅ Training complete. Model saved.")