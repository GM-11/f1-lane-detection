from torch.utils.data import Dataset
import os
import torchvision.transforms as T
from PIL import Image
import torch


class CULaneDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=(256, 512)):
        self.img_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        self.img_size = img_size

        self.image_files = sorted(os.listdir(self.img_dir))
        # only keep images that have corresponding masks
        self.image_files = [
            f
            for f in self.image_files
            if os.path.exists(
                os.path.join(self.mask_dir, f.replace(".jpg", "_mask.png"))
            )
        ]

        # transforms: resize + convert to tensor
        self.transform_img = T.Compose(
            [
                T.Resize(self.img_size),
                T.ToTensor(),  # [0,1]
            ]
        )
        self.transform_mask = T.Compose(
            [
                T.Resize(self.img_size, interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor(),  # [0,1]
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = img_name.replace(".jpg", "_mask.png")

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale

        image = self.transform_img(image)
        mask = self.transform_mask(mask)

        # ensure mask is binary (0 or 1)
        mask = (torch.tensor(mask) > 0.5).float()

        return image, mask
