import os

import cv2
import matplotlib.pyplot as plt
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from supervision.utils.conversion import pillow_to_cv2

mapping = {
    0: "PAI 3",
    1: "PAI 4",
    2: "PAI 5",
}

img_path = os.path.join(os.getcwd(), "data", "InferenceData", "inference_3.jpg")
image = Image.open(img_path)

model = RFDETRBase(pretrain_weights=os.path.join(os.getcwd(), "rf_detr", "checkpoint_best_total.pth"), num_classes=2)
detections = model.predict(image, threshold=0.25)

text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

bbox_annotator = sv.BoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(
    text_color=sv.Color.BLACK,
    text_scale=text_scale,
    text_thickness=thickness,
    smart_position=True)

detections_labels = [
    f"{mapping[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

detections_image = image.copy()
detections_image = bbox_annotator.annotate(detections_image, detections)
detections_image = label_annotator.annotate(detections_image, detections, detections_labels)

# Show image
if isinstance(detections_image, Image.Image):
    detections_image = pillow_to_cv2(detections_image)

plt.figure(figsize=(12, 12))

plt.imshow(cv2.cvtColor(detections_image, cv2.COLOR_BGR2RGB))

plt.axis("off")
plt.title("Annotated Image with RF-DETR")
plt.show()
