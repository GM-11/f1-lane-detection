import cv2
import sys
from detect_lanes_using_model import detect_lanes_and_draw_lines
from box_around_car import draw_box_around_car
import matplotlib.pyplot as plt
import numpy as np
video = cv2.VideoCapture("./data/lando_monaco_lap.mp4")


def main_with_cv2():

    if not video.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = video.read()

        if not ret:
                break

        frame_with_lines = detect_lanes_and_draw_lines(frame)
        frame_with_box = draw_box_around_car(frame_with_lines)

            # Display the frame
        cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Lane Detection", 1000, 750)
        cv2.imshow("Lane Detection", frame_with_box)

                # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


        frame_count += 1
        if frame_count % 30 == 0:  # Print progress every 30 frames
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")




        video.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames total")

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
