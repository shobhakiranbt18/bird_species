import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm   # For progress bar

# ============================
# CORRECT PATHS WITH DOUBLE SLASHES
# ============================
dataset_dir = r"C:\\Bird_Species\\dataset\\Train"


model_dir = r"C:\\Bird_Species\\model"
model_path = r"C:\\Bird_Species\\model\\bird_model.pth"

os.makedirs(model_dir, exist_ok=True)

# ============================
# TRANSFORMS (FAST CPU TRAINING)
# ============================
transform = transforms.Compose([
    transforms.Resize((128, 128)),   # Faster training
    transforms.ToTensor()
])

print("Loading dataset...")

# ============================
# LOAD DATASET
# ============================
train_dataset = datasets.ImageFolder(dataset_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

print("Dataset Loaded!")
print(f"Total Classes : {len(train_dataset.classes)}")
print(f"Total Images  : {len(train_dataset)}")

# ============================
# MODEL
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 5
print("\nTraining Started...\n")

# ============================
# TRAINING LOOP
# ============================
for epoch in range(epochs):
    running_loss = 0
    correct = 0
    total = 0

    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    print(f"Epoch {epoch+1} Completed | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%\n")
# ============================
# SAVE MODEL
# ============================
torch.save(model.state_dict(), model_path)

print("Training Completed!")
print(f"Model saved at: {model_path}")

# ============================
# SAVE CLASS NAMES
# ============================
import json

class_names_file = os.path.join(model_dir, "class_names.json")

with open(class_names_file, "w") as f:
    json.dump(train_dataset.classes, f)

print(f"Class names saved at: {class_names_file}")

