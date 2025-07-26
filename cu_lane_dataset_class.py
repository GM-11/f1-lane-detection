import os
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CULaneDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=(256, 512), transform=None):
        self.img_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        self.img_size = img_size
        self.transform = transform

        self.image_files = sorted(os.listdir(self.img_dir))
        self.image_files = [
            f
            for f in self.image_files
            if os.path.exists(
                os.path.join(self.mask_dir, f.replace(".jpg", "_mask.png"))
            )
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = img_name.replace(".jpg", "_mask.png")

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # use cv2 because albumentations expects numpy arrays
        image = cv2.imread(img_path)  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # single channel

        # apply albumentations transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            # fallback: just resize and to tensor
            basic_transform = A.Compose(
                [
                    A.Resize(self.img_size[0], self.img_size[1]),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
            augmented = basic_transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # ensure mask is binary [0,1] and add channel dim if missing
        mask = (mask > 0.5).float().unsqueeze(0) if mask.ndim == 2 else mask.float()

        return image, mask
