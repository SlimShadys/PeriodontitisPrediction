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

font = ImageFont.truetype("arial.ttf", 40)

model = YOLO(os.path.join(os.getcwd(), "runs", "detect", "v2.1", "drawn-glitter-15", "weights", "epoch_22.pt"))
img_path = os.path.join(os.getcwd(), "data", "InferenceData", "inference_1.jpg")

results = model.predict(img_path, imgsz=1280, conf=0.25, device="cuda",)

# Extract boxes from the results
boxes = results[0].boxes

teeth_boxes = []
for box in boxes:
    class_name = results[0].names[box.cls.cpu().numpy()[0]]
    teeth_boxes.append([class_name, [int(v) for v in box.xyxy.cpu().numpy()[0]]])

img = Image.open(img_path)
draw = ImageDraw.Draw(img)

for cls, box in teeth_boxes:
    draw.rectangle(box, outline=colors[int(cls)], width=2)
    # Convert the class to the corresponding class name
    if int(cls) in class_mapping:
        real_name = class_mapping[int(cls)]
    else:
        real_name = cls
    draw.text((box[0], box[1]), str(real_name), fill=colors[int(cls)], font=font)

img.show()
