import os

import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase

img_path = os.path.join(os.getcwd(), "data", "InferenceData", "inference_3.jpg")
image = Image.open(img_path)

model = RFDETRBase(pretrain_weights=os.path.join(os.getcwd(), "yolo_det", "best-lively-durian-49.pt"))
detections = model.predict(image, threshold=0.25)

# labels = [
#     f"{COCO_CLASSES[class_id]} {confidence:.2f}"
#     for class_id, confidence
#     in zip(detections.class_id, detections.confidence)
# ]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
# annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)
