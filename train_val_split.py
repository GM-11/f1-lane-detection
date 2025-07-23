import os
import shutil
import glob
import random

# Adjust these paths
IMAGE_ROOT = "./driver_161_90frame"  # where your MP4 folders with images live
MASK_ROOT = "./driver_161_90frame_masks"  # where your masks live
OUTPUT_ROOT = "./culane_split_dataset"  # output folder

# Desired split
TRAIN_RATIO = 0.8

# Collect all image paths
all_images = glob.glob(os.path.join(IMAGE_ROOT, "**", "*.jpg"), recursive=True)
print(f"Found {len(all_images)} images.")

# Shuffle for randomness
random.shuffle(all_images)

# Split
train_cutoff = int(len(all_images) * TRAIN_RATIO)
train_images = all_images[:train_cutoff]
val_images = all_images[train_cutoff:]


# Helper to copy
def copy_split(image_list, split_name):
    img_out_dir = os.path.join(OUTPUT_ROOT, split_name, "images")
    msk_out_dir = os.path.join(OUTPUT_ROOT, split_name, "masks")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(msk_out_dir, exist_ok=True)

    for img_path in image_list:
        # relative parts
        filename = os.path.basename(img_path)
        foldername = os.path.basename(os.path.dirname(img_path))
        # mask path
        mask_name = os.path.splitext(filename)[0] + "_mask.png"
        mask_path = os.path.join(MASK_ROOT, foldername, mask_name)

        if not os.path.exists(mask_path):
            print(f"[WARN] Missing mask for {img_path}")
            continue

        # destination
        img_dst = os.path.join(img_out_dir, f"{foldername}_{filename}")
        msk_dst = os.path.join(msk_out_dir, f"{foldername}_{mask_name}")

        shutil.copy2(img_path, img_dst)
        shutil.copy2(mask_path, msk_dst)

    print(f"[INFO] {split_name} done: {len(image_list)} images")


# Run copy
copy_split(train_images, "train")
copy_split(val_images, "val")
