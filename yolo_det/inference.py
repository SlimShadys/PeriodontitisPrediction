from ultralytics import YOLO
import os
from PIL import Image, ImageDraw, ImageFont

# Specify a color for each of the 12 classes
# 0-3: Quadrants
# 4-11: Teeth
colors = {
    0: 'red',
    1: 'green',
    2: 'blue',
    3: 'yellow',
    4: 'purple',
    5: 'orange',
    6: 'cyan',
    7: 'magenta',
    8: 'lime',
    9: 'pink',
    10: 'teal',
    11: 'lavender'
}

class_mapping = {
    0: 2, # Class 0 is Quadrant 2
    1: 1, # Class 1 is Quadrant 1
    2: 3, # Class 2 is Quadrant 3
    3: 4, # Class 3 is Quadrant 4
}

class_mapping = {
    0: 'PAI 3', # Class 0 PAI 3
    1: 'PAI 4', # Class 1 PAI 4
    2: 'PAI 5', # Class 2 PAI 5
}

font = ImageFont.truetype("arial.ttf", 40)

# model = YOLO(os.path.join(os.getcwd(), "runs", "detect", "v2.1", "drawn-glitter-15", "weights", "epoch_22.pt"))
model = YOLO(os.path.join(os.getcwd(), "yolo_det", "best-lively-durian-49.pt"))

img_path = os.path.join(os.getcwd(), "data", "InferenceData", "inference_3.jpg")

results = model.predict(img_path, imgsz=1280, conf=0.25, device="cuda",)

# Extract boxes from the results
boxes = results[0].boxes

teeth_boxes = []
for box in boxes:
    class_yolo = box.cls.cpu().numpy()[0]
    class_name = results[0].names[class_yolo]
    teeth_boxes.append([class_yolo, class_name, [int(v) for v in box.xyxy.cpu().numpy()[0]]])

img = Image.open(img_path)
draw = ImageDraw.Draw(img)

for yolo_cls, name_cls, box in teeth_boxes:
    draw.rectangle(box, outline=colors[int(yolo_cls)], width=2)
    
    # Convert the class to the corresponding class name
    draw.text((box[0], box[1]+100), str(name_cls), fill=colors[int(yolo_cls)], font=font)

img.show()
