import cv2
import sys
from detect_lanes_using_model import detect_lanes_and_draw_lines
from box_around_car import draw_box_around_car
import matplotlib.pyplot as plt
import numpy as np
# video = cv2.VideoCapture("./data/lando_monaco_lap.mp4")
# video = cv2.VideoCapture("./data/lewis_monza_lap.mp4")
video = cv2.VideoCapture("./data/oscar_bahrain_pole.mp4")


def main_with_plt():

    if not video.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Set up the plot
    plt.ion()  # Turn on interactive mode
    img_plot = plt.imshow(np.zeros((720, 1280, 3), dtype=np.uint8))
    plt.axis('off')  # Hide axes

    while True:
        ret, frame = video.read()

        if not ret:
                break

        frame_with_lines = detect_lanes_and_draw_lines(frame)
        frame_with_box = draw_box_around_car(frame_with_lines)

        # Convert BGR to RGB for matplotlib
        rgb_frame = cv2.cvtColor(frame_with_box, cv2.COLOR_BGR2RGB)

        # Update the plot with new frame
        img_plot.set_data(rgb_frame)
        plt.draw()
        plt.pause(1/fps)  # Pause to create animation effect

    # Clean up
    video.release()
    plt.ioff()
    plt.close()

if __name__ == "__main__":
    main_with_plt()
