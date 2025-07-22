import cv2
import numpy as np
from detect_lanes import detect_lanes_and_draw_lines

video = cv2.VideoCapture("./data/converted_spa_onboard.mp4")


if not video.isOpened():
    print("Error: Could not open video.")
else:
    while True:
        ret, frame = video.read()

        if not ret:
            break
        frame_with_lines = detect_lanes_and_draw_lines(frame)

        cv2.imshow("Lane Detection", frame_with_lines)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    video.release()
    cv2.destroyAllWindows()
