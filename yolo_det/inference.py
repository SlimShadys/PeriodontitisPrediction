import os

import cv2
import matplotlib.pyplot as plt
import supervision as sv
from PIL import Image
from supervision.utils.conversion import pillow_to_cv2
from ultralytics import YOLO

# Class label mapping
class_mapping = {
    0: 'PAI 3',
    1: 'PAI 4',
    2: 'PAI 5',
}

# Load YOLO model
model = YOLO(os.path.join(os.getcwd(), "yolo_det", "best-lively-durian-49.pt"))

# Load image
img_path = os.path.join(os.getcwd(), "data", "InferenceData", "inference_1.jpg")
image = Image.open(img_path)
image_width, image_height = image.size

# Run prediction
results = model.predict(img_path, imgsz=1280, conf=0.25, device="cuda")

# Extract boxes, class_ids, and confidences
boxes = results[0].boxes
xyxy = boxes.xyxy.cpu().numpy()
confidences = boxes.conf.cpu().numpy()
class_ids = boxes.cls.cpu().numpy().astype(int)

# Create Detections object
detections = sv.Detections(
    xyxy=xyxy,
    confidence=confidences,
    class_id=class_ids
)

# Setup annotators
text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

bbox_annotator = sv.BoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(
    text_color=sv.Color.BLACK,
    text_scale=text_scale,
    text_thickness=thickness,
    smart_position=True
)

# Create detection labels
detection_labels = [
    f"{class_mapping.get(class_id, str(class_id))} {confidence:.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
]

# Annotate image
annotated_image = image.copy()
annotated_image = bbox_annotator.annotate(annotated_image, detections)
annotated_image = label_annotator.annotate(annotated_image, detections, detection_labels)

# Show image
if isinstance(annotated_image, Image.Image):
    detections_image = pillow_to_cv2(annotated_image)

plt.figure(figsize=(12, 12))

plt.imshow(cv2.cvtColor(detections_image, cv2.COLOR_BGR2RGB))

plt.axis("off")
plt.title("Annotated Image with YOLO")
plt.show()
