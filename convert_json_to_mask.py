# %%
import os
import json
import numpy as np
from labelme import utils
import PIL.Image

def convert_json_to_mask(json_path, out_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    img = utils.img_b64_to_arr(data['imageData'])
    label_name_to_value = {'_background_': 0}

    for shape in data['shapes']:
        label_name = shape['label']
        if label_name not in label_name_to_value:
            label_name_to_value[label_name] = len(label_name_to_value)

    lbl, _ = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

    # Binarize the mask: 0 = background, 255 = track
    bin_mask = (lbl > 0).astype(np.uint8) * 255

    out_file = os.path.join(out_dir, os.path.splitext(os.path.basename(json_path))[0] + "_mask.png")
    PIL.Image.fromarray(bin_mask).save(out_file)
    print(f"Saved: {out_file}")

for i in range(67, 68):
    json_path = f"new_data/frame_{i:04d}.json"
    out_dir = "new_data/masks/"
    os.makedirs(out_dir, exist_ok=True)
    convert_json_to_mask(json_path, out_dir)


# %%
import shutil
import os
mask_dir = "new_data/masks/"
os.makedirs(mask_dir, exist_ok=True)

for filename in os.listdir("new_data/images"):
    if filename.endswith("_mask.png"):
        source_path = os.path.join("new_data/images", filename)
        dest_path = os.path.join(mask_dir, filename)
        shutil.move(source_path, dest_path)
        print(f"Moved {filename} to {mask_dir}")

# %%
frame_json_dir = "new_data/frames_json/"
os.makedirs(frame_json_dir, exist_ok=True)

for filename in os.listdir("new_data"):
    if filename.endswith(".json"):
        source_path = os.path.join("new_data", filename)
        dest_path = os.path.join(frame_json_dir, filename)
        shutil.move(source_path, dest_path)
        print(f"Moved {filename} to {frame_json_dir}")
