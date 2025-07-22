import cv2
import numpy as np


def road_region(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(img=mask, pts=[vertices], color=(255, 255, 255))
    print(image, mask)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def draw_lines(image, lines, color=(0, 255, 0), thickness=3):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle_deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle_deg) > 180 or abs(angle_deg) == 0:
                continue
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


def detect_lanes_and_draw_lines(frame):
    # Convert frame to grayscale to make lanes come out due to contrast
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply guassian blur to reduce the noise in image
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    # detect canny edges
    edges = cv2.Canny(blurred_frame, 100, 110)

    # Define the region of interest (road region)
    height, width = edges.shape
    vertices = np.array(
        [
            [
                (0, height * 0.4),  # bottom left corner
                (width * 0.3, height * 0.22),  # top left corner
                (width * 0.7, height * 0.22),  # top right corner
                (width * 0.9, height * 0.4),  # bottom right corner
                (width * 0.9, height * 0.9),  # bottom right corner
                (width * 0.8, height * 0.4),  # bottom right corner
                (width * 0.2, height * 0.4),  # bottom right corner
                (width * 0.1, height * 0.9),  # bottom right corner
            ]
        ],
        dtype=np.int32,
    )
    masked_edges = road_region(edges, vertices)
    print(masked_edges)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [vertices], (0, 255, 255))
    cv2.imshow("ROI Mask", mask)  # Show the trapezoid mask
    cv2.imshow("Edges", edges)  # Show raw edges from Canny
    cv2.imshow("Masked Edges", masked_edges)
    # find the lines
    lines = cv2.HoughLinesP(
        masked_edges,
        2,
        np.pi / 180,
        threshold=50,
        minLineLength=10,
        maxLineGap=250,
    )
    print(lines)

    frame_with_lines = draw_lines(frame, lines)

    return frame_with_lines
