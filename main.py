import cv2
from detect_lanes import detect_lanes_and_draw_lines
from box_around_car import draw_box_around_car

video = cv2.VideoCapture("./data/converted_spa_onboard.mp4")


if not video.isOpened():
    print("Error: Could not open video.")
else:
    while True:
        ret, frame = video.read()

        if not ret:
            break
        frame_with_lines = detect_lanes_and_draw_lines(frame)
        frame_with_box = draw_box_around_car(frame_with_lines)

        cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Lane Detection", 1000, 750)
        cv2.imshow("Lane Detection", frame_with_box)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    video.release()
    cv2.destroyAllWindows()
