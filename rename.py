import os
import re
import shutil

# Get all PNG files in the new_data folder
folder_path = "new_data"
png_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]

# Extract frame numbers using regex
frame_pattern = re.compile(r"frame_(\d+)\.png")
frame_numbers = []
frame_files = {}

for file in png_files:
    match = frame_pattern.match(file)
    if match:
        frame_num = int(match.group(1))
        frame_numbers.append(frame_num)
        frame_files[frame_num] = file

# Sort frame numbers
frame_numbers.sort()

# Rename files to continuous numbering
for new_idx, old_frame_num in enumerate(frame_numbers):
    old_filename = frame_files[old_frame_num]
    new_filename = f"frame_{new_idx:04d}.png"

    # Only rename if the filename would change
    if old_filename != new_filename:
        old_path = os.path.join(folder_path, old_filename)
        new_path = os.path.join(folder_path, new_filename)
        print(f"Renaming {old_filename} to {new_filename}")
        shutil.move(old_path, new_path)
