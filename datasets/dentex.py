import json
import os
import random
import shutil
from collections import defaultdict

import yaml
from tqdm import tqdm

class Dentex():
    def __init__(self, dataset_configs):
        # Generic variables
        self.dataset_name = dataset_configs["name"]
        self.data_path = dataset_configs["path"]
        self.create_yolo_version = dataset_configs["create_yolo_version"] # If True, the dataset will be converted to the YOLO format

        # Directories
        self.train_dir = os.path.join(self.data_path, 'training_data')
        self.val_dir = os.path.join(self.data_path, 'validation_data')
        self.test_dir = os.path.join(self.data_path, 'test_data')

        # Variants and JSON files for each variant
        self.dataset_variants = ['quadrant', 'quadrant_enumeration']
        self.quadrant_file = os.path.join(self.train_dir, "quadrant", "train_quadrant.json")
        self.quadrant_enum_file = os.path.join(self.train_dir, "quadrant_enumeration", "train_quadrant_enumeration.json")

        # Get quadrant data
        with open(self.quadrant_file, 'r') as f:
            self.quadrant_json = json.load(f)

        # Get quadrant enumeration data
        with open(self.quadrant_enum_file, 'r') as f:
            self.quadrant_enum_json = json.load(f)
        
        # Create the YOLO dataset only if the flag is set to True
        if self.create_yolo_version:

            # Output directory for YOLO labels (create if necessary)
            self.yolo_output_dir = os.path.join(self.data_path, "YOLO_dataset")

            if not os.path.exists(os.path.join(self.yolo_output_dir, "train")):
                os.makedirs(os.path.join(self.yolo_output_dir, "train"), exist_ok=True)
            if not os.path.exists(os.path.join(self.yolo_output_dir, "val")):
                os.makedirs(os.path.join(self.yolo_output_dir, "val"), exist_ok=True)

            # We'd like to extract the two categories (quadrant and quadrant+enumeration) and unify them
            categories_1 = self.quadrant_json["categories"]
            categories_2 = self.quadrant_enum_json["categories_2"]

            class_map, quadrant_classes, tooth_classes = self.unify_categories(categories_1, categories_2)

            # Process quadrant data
            quadrant_data = self.convert_to_yolo_format(self.quadrant_json, quadrant_classes)
            self.process_yolo_data(quadrant_data, os.path.join(self.train_dir, "quadrant", "xrays"))

            # Process quadrant enumeration data with offset
            quadrant_enum_data = self.convert_to_yolo_format(self.quadrant_enum_json, quadrant_classes)
            len_quadrants = len(os.listdir(os.path.join(self.train_dir, "quadrant", "xrays")))
            self.process_yolo_data(quadrant_enum_data, os.path.join(self.train_dir, "quadrant_enumeration", "xrays"), len_offset=len_quadrants)

            # Here there could be a re-mapping of the validation IDs to start from a 0-based index

            # Create the data.yaml file
            self.create_yaml(self.yolo_output_dir, class_map)

            print("YOLO dataset and data.yaml successfully generated!")

    def unify_categories(self, categories_1, categories_2):
        quadrant_classes = {c["id"]: f"q{c['name']}" for c in categories_1}
        tooth_classes = {c["id"]: f"t{c['name']}" for c in categories_2}
        
        # Offset tooth IDs to avoid overriding quadrant IDs
        num_quadrants = len(quadrant_classes)
        tooth_classes_offset = {k + num_quadrants: v for k, v in tooth_classes.items()}

        # Merge both dictionaries properly
        merged_classes = {**quadrant_classes, **tooth_classes_offset}

        return merged_classes, quadrant_classes, tooth_classes_offset

    # Start the process of converting the Dentex dataset to the YOLO format
    def process_yolo_data(self, data, src_dir, len_offset=0):
        random.seed(42)
        val_size = 0.2
        # train_size = 1 - val_size

        image_keys = list(data.keys())
        random.shuffle(image_keys)

        num_val_images = int(val_size * len(image_keys))
        train_images = image_keys[num_val_images:]
        val_images = image_keys[:num_val_images]

        desc = "Processing YOLO data"
        desc += " (validation)" if len_offset > 0 else " (training)"

        for orig_file_name, labels in tqdm(data.items(), desc=desc, unit="images"):
            if orig_file_name in val_images:
                output_subdir = 'val'
                file_name = orig_file_name.replace("train_", "val_")
            elif orig_file_name in train_images:
                output_subdir = 'train'
                file_name = orig_file_name
            else:
                raise ValueError(f"Image {orig_file_name} not found in training or validation set.")

            # Adjust file name with offset if needed
            if len_offset > 0:
                base_name, ext = file_name.split(".")
                prefix, idx = base_name.split("_")
                file_name = f"{prefix}_{int(idx) + len_offset}.{ext}"

            # Write YOLO labels to files
            txt_path = os.path.join(self.yolo_output_dir, output_subdir, file_name.replace(".png", ".txt"))
            with open(txt_path, 'w') as f:
                f.write("\n".join(labels))

            # Copy the image to the YOLO output directory
            shutil.copy(os.path.join(src_dir, orig_file_name), os.path.join(self.yolo_output_dir, output_subdir, file_name))

    def convert_to_yolo_format(self, json_data, quadrant_classes):
        """
        Converts the dataset to YOLO format for training or validation.

        Args:
            json_data (dict): The JSON data containing annotations and images.
            quadrant_classes (dict): Mapping of quadrant class IDs to names.
        """
        image_dict = {img["id"]: img for img in json_data["images"]}
        yolo_labels = defaultdict(list)

        for ann in json_data["annotations"]:
            img_id = ann["image_id"]
            
            img_info = image_dict[img_id]
            img_width, img_height = img_info["width"], img_info["height"]

            # Extract bounding box coordinates
            x, y, w, h = ann["bbox"]

            # Convert to YOLO format
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            width = w / img_width
            height = h / img_height

            # Determine class ID (quadrant or tooth)
            # If quadrant, leave the class ID as is
            # If tooth, add an offset to the class ID to avoid overriding quadrant IDs
            if "category_id_1" in ann and "category_id_2" in ann:
                class_id = ann["category_id_2"] + len(quadrant_classes) # Tooth class ID offset
            else:
                class_id = ann["category_id"] # Quadrant class ID

            yolo_labels[img_info["file_name"]].append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return yolo_labels

    def create_yaml(self, yolo_dir, class_map):
        yaml_content = {
            "train": os.path.join(os.getcwd(), self.yolo_output_dir, "train"),
            "val": os.path.join(os.getcwd(), self.yolo_output_dir, "val"),
            "nc": len(class_map),
            "names": list(class_map.keys())
        }
        
        with open(os.path.join(yolo_dir, "data.yaml"), 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)