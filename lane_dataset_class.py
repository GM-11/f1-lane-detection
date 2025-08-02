import os
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class LaneDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=(256, 512), transform=None):

        self.img_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        self.img_size = img_size
        self.transform = transform

        self.image_files = []
        for dirpath, _, filenames in os.walk(self.img_dir):
            for fname in filenames:
                if fname.lower().endswith(".png"):
                    rel_path = os.path.relpath(
                        os.path.join(dirpath, fname), self.img_dir
                    )
                    mask_rel_path = rel_path.replace(".png", "_mask.png")
                    mask_full_path = os.path.join(self.mask_dir, mask_rel_path)
                    if os.path.exists(mask_full_path):
                        self.image_files.append(rel_path)

        self.image_files.sort()  # optional, for consistency

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        rel_path = self.image_files[idx]

        img_path = os.path.join(self.img_dir, rel_path)
        mask_path = os.path.join(self.mask_dir, rel_path.replace(".png", "_mask.png"))

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask at {mask_path}")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
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

        if mask.ndim == 2:  # (H, W)
            mask = (mask > 0.5).float().unsqueeze(0)
        else:
            mask = (mask > 0.5).float()

        return image, mask
