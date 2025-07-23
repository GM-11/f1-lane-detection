# %%cell
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cu_lane_dataset_class import CULaneDataset

# %%cell 2
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %%cell 3


# %%cell 4
train_dataset = CULaneDataset(
    "./culane_split_dataset", split="train", img_size=(256, 512)
)
val_dataset = CULaneDataset("./culane_split_dataset", split="val", img_size=(256, 512))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)


# %%cell 5


class LaneDetectionCNN(nn.Module):
    def __init__(self):
        super(LaneDetectionCNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),  # output
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)  # apply sigmoid to get probabilities


# %%cell 6
model = LaneDetectionCNN()
epochs = 1
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model = model.to(device)
# %%cell 7
for i in range(epochs):
    total_epoch_loss = 0

    for b_features, b_labels in train_loader:
        b_features = b_features.to(device)
        b_labels = b_labels.to(device)

        # Forward pass
        y_pred = model(b_features)
        loss = criterion(y_pred, b_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Batch Loss: {loss.item():.4f}")
        total_epoch_loss += loss.item()

    avg_loss = total_epoch_loss / len(train_loader)
    print(f"Epoch {i + 1}/{epochs}, Loss: {avg_loss:.4f}")

# %%cell 8
total = 0
correct = 0

with torch.no_grad():
    for batch_features, batch_labels in val_loader:
        # move data to gpu
        batch_features, batch_labels = (
            batch_features.to(device),
            batch_labels.to(device),
        )

        outputs = model(batch_features)

        _, predicted = torch.max(outputs, 1)

        total = total + batch_labels.shape[0]

        correct = correct + (predicted == batch_labels).sum().item()

print(correct / total)
