import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model_defination import LaneDetectionUNet  # Assuming this is the correct import path

model = LaneDetectionUNet()
checkpoint = torch.load("f1_lane_detection.pth", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint)
model.eval()


IMG_HEIGHT, IMG_WIDTH = 256, 512
threshold = 0.4

preprocess = A.Compose(
    [
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_image = image.copy()

    transformed = preprocess(image=image)
    processed_image = transformed["image"]

    processed_image = processed_image.unsqueeze(0)

    return processed_image, original_image


def postprocess_output(output):
    """Convert model output to binary mask."""
    output = torch.sigmoid(output)
    binary_mask = (output >= threshold).float()
    binary_mask = binary_mask.squeeze().cpu().numpy()
    return binary_mask



def predict_lanes(image_path):
    """Complete pipeline to predict lanes on an image."""
    try:
        processed_image, original_image = load_and_preprocess_image(image_path)

        with torch.no_grad():
            output = model(processed_image)

        lane_mask = postprocess_output(output)

        print(f"Lane detection completed for {image_path}")
        return lane_mask, original_image

    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None


def simple_inference(image_path):
    """Simple inference function that returns just the lane mask."""
    try:
        processed_image, _ = load_and_preprocess_image(image_path)

        with torch.no_grad():
            output = model(processed_image)

        lane_mask = postprocess_output(output)
        return lane_mask
    except Exception as e:
        print(f"Error in inference: {e}")
        return None


def preprocess_frame(frame):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    transformed = preprocess(image=frame_rgb)
    processed_frame = transformed["image"]

    processed_frame = processed_frame.unsqueeze(0)

    return processed_frame, frame_rgb


def draw_lane_lines_on_frame(original_frame, lane_mask):

    frame_height, frame_width = original_frame.shape[:2]
    lane_mask_resized = cv2.resize(lane_mask.astype(np.uint8), (frame_width, frame_height))


    overlay = np.zeros_like(original_frame)
    overlay[lane_mask_resized > 0] = [0, 255, 0]  # Green color for lanes


    alpha = 0.3
    frame_with_lanes = cv2.addWeighted(original_frame, 1 - alpha, overlay, alpha, 0)

    return frame_with_lanes


def detect_lanes_and_draw_lines(frame):
    try:
        processed_frame, frame_rgb = preprocess_frame(frame)

        with torch.no_grad():
            output = model(processed_frame)

        lane_mask = postprocess_output(output)

        frame_with_lines = draw_lane_lines_on_frame(frame, lane_mask)

        return frame_with_lines

    except Exception as e:
        print(f"Error in lane detection: {e}")
        return frame
