import os
import json
from PIL import Image

def convert_yolo_to_coco(yolo_dir, output_json, class_names):
    image_id = 0
    annotation_id = 0
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Add categories
    for idx, class_name in enumerate(class_names):
        coco_data["categories"].append({
            "id": idx,
            "name": class_name,
            "supercategory": "none"
        })

    for filename in sorted(os.listdir(yolo_dir)):
        if not filename.endswith('.txt'):
            continue

        image_filename = filename.replace('.txt', '.jpg')
        image_path = os.path.join(yolo_dir, image_filename)
        txt_path = os.path.join(yolo_dir, filename)

        if not os.path.exists(image_path):
            continue  # skip if corresponding image not found

        # Load image for dimensions
        with Image.open(image_path) as img:
            width, height = img.size

        # Add image info
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        })

        # Parse annotation
        with open(txt_path, "r") as f:
            for line in f:
                if line.strip() == "":
                    continue
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:])

                # Convert from YOLO normalized format to COCO absolute
                x = (x_center - w / 2) * width
                y = (y_center - h / 2) * height
                abs_w = w * width
                abs_h = h * height

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x, y, abs_w, abs_h],
                    "area": abs_w * abs_h,
                    "iscrowd": 0
                })
                annotation_id += 1

        image_id += 1

    # Save to COCO json
    with open(output_json, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"Saved COCO annotations to: {output_json}")

# Example usage
class_names = ["PAI 1", "PAI 2", "PAI 3"]  # Replace with your actual class names

convert_yolo_to_coco("data/Periapical Dataset/Periapical Lesions/YOLO_dataset/train", "data/Periapical Dataset/Periapical Lesions/YOLO_dataset/train/_annotations.coco.json", class_names)
convert_yolo_to_coco("data/Periapical Dataset/Periapical Lesions/YOLO_dataset/val", "data/Periapical Dataset/Periapical Lesions/YOLO_dataset/val/_annotations.coco.json", class_names)
