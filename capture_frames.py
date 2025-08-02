import cv2
import os

saved_count = 22


def extract_frames(video_path, output_folder, interval_sec=1):
    global saved_count  # Use the global variable

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get the frame rate (frames per second)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:04d}.png")
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            saved_count += 1

        frame_count += 1

    cap.release()
    print("Done.")


# Example usage
# video_path = "data/converted_spa_onboard.mp4"  # Change this to your video path
output_folder = "new_data"
# extract_frames(video_path, output_folder, interval_sec=5)

for v in [
    "data/lando_monaco_pole.mp4",
    "data/max_silverstone_pole.mp4",
    "data/lewis_chinese_sprint_pole.mp4",
]:
    video_path = v
    print(f"Processing video: {video_path}")
    extract_frames(video_path, output_folder, interval_sec=1)
    print(f"Frames extracted from {video_path} to {output_folder}")
