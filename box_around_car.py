import cv2
from cv2.typing import MatLike


def draw_box_around_car(frame: MatLike) -> MatLike:
    print(frame.shape)
    width, height, _ = frame.shape

    # Draw a rectangle around the car
    top_left = (int(width * 0.2), int(height * 0.22))
    bottom_right = (int(width * 1.75), int(height))
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)  # Green box
    return frame
