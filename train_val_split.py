import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# ===== CONFIG =====
input_dir = "new_data"
train_ratio = 0.8
output_dir = "f1_road_dataset"  # will create train/ and test/ inside
random.seed(42)
# ==================

def collect_image_mask_pairs():
    """Collect all image-mask pairs from the input directory"""
    pairs = []

    images_dir = os.path.join(input_dir, "images")
    masks_dir = os.path.join(input_dir, "masks")

    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"[ERROR] Images or masks directory not found in {input_dir}")
        return pairs

    # Get all image files
    for file in os.listdir(images_dir):
        if file.endswith('.png'):
            img_path = os.path.join(images_dir, file)

            # Look for corresponding mask file
            mask_name = file.replace('.png', '_mask.png')
            mask_path = os.path.join(masks_dir, mask_name)

            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
            else:
                print(f"[WARN] No mask found for {img_path}")

    return pairs

def copy_files(pairs, split_name):
    """Copy image-mask pairs to the appropriate split directory"""
    for img_path, mask_path in tqdm(pairs, desc=f"Copying {split_name}"):
        # Get filenames
        img_filename = os.path.basename(img_path)
        mask_filename = os.path.basename(mask_path)

        # Create output paths
        out_img_path = os.path.join(output_dir, split_name, "images", img_filename)
        out_mask_path = os.path.join(output_dir, split_name, "masks", mask_filename)

        # Create directories
        Path(out_img_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_mask_path).parent.mkdir(parents=True, exist_ok=True)

        # Copy files
        shutil.copy2(img_path, out_img_path)
        shutil.copy2(mask_path, out_mask_path)

# ===== Main execution =====
print("Collecting image-mask pairs...")
pairs = collect_image_mask_pairs()

if not pairs:
    print("No image-mask pairs found!")
    exit(1)

# Shuffle and split
random.shuffle(pairs)
split_idx = int(len(pairs) * train_ratio)
train_pairs = pairs[:split_idx]
test_pairs = pairs[split_idx:]

print(f"Total pairs: {len(pairs)}")
print(f"Train pairs: {len(train_pairs)}")
print(f"Test pairs: {len(test_pairs)}")

# Create output directories
os.makedirs(os.path.join(output_dir, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "train", "masks"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test", "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test", "masks"), exist_ok=True)

# Copy files
copy_files(train_pairs, "train")
copy_files(test_pairs, "test")

print("✅ Done! Dataset split completed at:", output_dir)
