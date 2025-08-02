# %%cell 1
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from torch.utils.data import Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from lane_dataset_class import LaneDataset

random.seed(42)
IMG_HEIGHT, IMG_WIDTH = 256, 512

# %%cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%cell 3
train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=5, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.3),
        A.GaussNoise(p=0.2),
        A.Resize(256, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

train_dataset =  LaneDataset(
    "/f1-road-dataset/f1_road_dataset",
    split="train",
    img_size=(256, 512),
    transform=train_transform,
)
val_dataset = LaneDataset(
    "/f1-road-dataset/f1_road_dataset", split="test", img_size=(256, 512), transform=val_transform
)


# Create data loaders with the subset for training
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
len(train_loader), len(val_loader)

# %%cell 6
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

# %%cell 5
class LaneDetectionUNet(nn.Module):
    def __init__(self):
        super(LaneDetectionUNet, self).__init__()

        # encoder
        self.down1 = DoubleConv(3, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.down2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.down3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.down4 = DoubleConv(64, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.down5 = DoubleConv(128, 256)
        self.pool5 = nn.MaxPool2d(kernel_size=2)

        self.down6 = DoubleConv(256, 512)
        self.pool6 = nn.MaxPool2d(kernel_size=2)

        # bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # decoder
        self.up_conv6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_conv6 = DoubleConv(1024, 512)

        self.up_conv5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv5 = DoubleConv(512, 256)

        self.up_conv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv4 = DoubleConv(256, 128)

        self.up_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv3 = DoubleConv(128, 64)

        self.up_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv2 = DoubleConv(64, 32)

        self.up_conv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec_conv1 = DoubleConv(32, 16)

        # final output layer
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # encoder
        x1 = self.down1(x)
        p1 = self.pool1(x1)

        x2 = self.down2(p1)
        p2 = self.pool2(x2)

        x3 = self.down3(p2)
        p3 = self.pool3(x3)

        x4 = self.down4(p3)
        p4 = self.pool4(x4)

        x5 = self.down5(p4)
        p5 = self.pool5(x5)

        x6 = self.down6(p5)
        p6 = self.pool6(x6)

        # bottleneck
        bottleneck = self.bottleneck(p6)

        # decoder
        up6 = self.up_conv6(bottleneck)
        x6 = torch.cat([up6, x6], dim=1)
        x6 = self.dec_conv6(x6)

        up5 = self.up_conv5(x6)
        x5 = torch.cat([up5, x5], dim=1)
        x5 = self.dec_conv5(x5)

        up4 = self.up_conv4(x5)
        x4 = torch.cat([up4, x4], dim=1)
        x4 = self.dec_conv4(x4)

        up3 = self.up_conv3(x4)
        x3 = torch.cat([up3, x3], dim=1)
        x3 = self.dec_conv3(x3)

        up2 = self.up_conv2(x3)
        x2 = torch.cat([up2, x2], dim=1)
        x2 = self.dec_conv2(x2)

        up1 = self.up_conv1(x2)
        x1 = torch.cat([up1, x1], dim=1)
        x1 = self.dec_conv1(x1)

        # final output
        return self.final_conv(x1)

# %%cell 5
model = LaneDetectionUNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model = model.to(device)


# %%cell 6
epochs = 30
for i in range(epochs):
    total_epoch_loss = 0.0
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = (
            batch_features.to(device),
            batch_labels.to(device),
        )

        y_pred = model(batch_features)
        loss = criterion(y_pred, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Batch Loss: {loss.item():.4f}")
        total_epoch_loss += loss.item()

    print(f"Epoch {i + 1}/{epochs}, Loss: {total_epoch_loss / len(train_loader):.4f}")

# %%cell 6
iou_scores = []
accuracy_scores = []

with torch.no_grad():
    for batch_features, batch_labels in val_loader:
        batch_features, batch_labels = (
            batch_features.to(device),
            batch_labels.to(device),
        )

        outputs = model(batch_features)

        preds = torch.sigmoid(outputs) >= 0.5

        intersection = (preds & (batch_labels.bool())).float().sum((1, 2, 3))
        union = (preds | (batch_labels.bool())).float().sum((1, 2, 3))

        union = torch.clamp(union, min=1e-6)
        batch_iou = (intersection / union).mean().item()
        iou_scores.append(batch_iou)

        correct_pixels = (preds == batch_labels.bool()).float().sum((1, 2, 3))
        total_pixels = (
            batch_labels.shape[1] * batch_labels.shape[2] * batch_labels.shape[3]
        )
        batch_accuracy = (correct_pixels / total_pixels).mean().item()
        accuracy_scores.append(batch_accuracy)

# Calculate average IoU and accuracy
mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
mean_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0

print(f"Validation IoU: {mean_iou:.4f}")
print(f"Validation Accuracy: {mean_accuracy:.4f}")
