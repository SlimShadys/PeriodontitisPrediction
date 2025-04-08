import json
import os
import random
import shutil
import sys
from collections import defaultdict

import cv2
import yaml
from tqdm import tqdm

# Local imports
sys.path.append("./")
from misc.augmentations import preprocess_image

# https://huggingface.co/datasets/ibrahimhamamci/DENTEX
class Dentex():
    def __init__(self, dataset_configs):
        # Generic variables
        self.dataset_name = dataset_configs["name"]
        self.data_path = dataset_configs["path"]
        self.create_yolo_version = dataset_configs["create_yolo_version"] # If True, the dataset will be converted to the YOLO format
        self.enhance_images = dataset_configs["enhance_images"] # If True, the images will be enhanced using the preprocess_image function

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

            class_map, _, _ = self.unify_categories(categories_1, categories_2)

            # Process quadrant data
            quadrant_data = self.convert_to_yolo_format(self.quadrant_json)
            
            # Process quadrant enumeration data
            quadrant_enum_data = self.convert_to_yolo_format(self.quadrant_enum_json)
            
            final_data = {
                "quadrant": [quadrant_data, os.path.join(self.train_dir, "quadrant", "xrays")],
                "quadrant_enumeration": [quadrant_enum_data, os.path.join(self.train_dir, "quadrant_enumeration", "xrays")],
                #"len_quadrants": len_quadrants
            }
            
            # Process the final data
            self.process_yolo_data(final_data)

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
    def process_yolo_data(self, data):
        # Extract data       
        # These are the quadrant and quadrant enumeration data
        # They are extracted as a dictionary with the following structure:
        # {
        #     "quadrant": [quadrant_data, src_quadrant],
        #     "quadrant_enumeration": [quadrant_enum_data, src_quadrant_enum]
        # }
        quadrant, src_quadrant = data["quadrant"]
        quadrant_enum, src_quadrant_enum = data["quadrant_enumeration"]
        
        # Seed and validation sizes
        random.seed(2047315)
        val_size = 0.2

        # Create a merged dataset empty dictionary
        merged_data = {}

        # Process quadrant data (classes 0-3)
        quadrant_keys = list(quadrant.keys())
        random.shuffle(quadrant_keys)
        num_val_quadrant = int(val_size * len(quadrant_keys))
        #quadrant_train_keys = quadrant_keys[num_val_quadrant:]
        quadrant_val_keys = quadrant_keys[:num_val_quadrant]

        # Process quadrant_enum data (classes 4-11)
        quadrant_enum_keys = list(quadrant_enum.keys())
        random.shuffle(quadrant_enum_keys)
        num_val_quadrant_enum = int(val_size * len(quadrant_enum_keys))
        #quadrant_enum_train_keys = quadrant_enum_keys[num_val_quadrant_enum:]
        quadrant_enum_val_keys = quadrant_enum_keys[:num_val_quadrant_enum]

        # Process quadrant data (classes 0-3)
        for orig_file_name in quadrant_keys:
            is_val = orig_file_name in quadrant_val_keys
            merged_data[orig_file_name] = {
                'labels': quadrant[orig_file_name],
                'src_dir': src_quadrant,
                'original_filename': orig_file_name,
                'is_val': is_val
            }
            
        # Process quadrant_enum data
        len_quadrant = len(quadrant)
        for orig_file_name in quadrant_enum_keys:
            # Rename the file by adding an offset to prevent overlap
            base_name, ext = orig_file_name.split(".")
            prefix, idx = base_name.split("_")
            new_file_name = f"{prefix}_{int(idx) + len_quadrant}.{ext}"

            # IMPORTANT: Modify quadrant_enum data labels to add class offset
            # From 0 to 3 (quadrant) and from 4 to 11 (enumeration)
            offset_labels = [
                f"{int(label.split()[0]) + 4} {' '.join(label.split()[1:])}" 
                for label in quadrant_enum[orig_file_name]
            ]

            is_val = orig_file_name in quadrant_enum_val_keys
            merged_data[new_file_name] = {
                'labels': offset_labels,
                'src_dir': src_quadrant_enum,
                'original_filename': orig_file_name,
                'is_val': is_val
            }

        # Plot statistics of the dataset (len of train and val sets)
        print('Image Dataset statistics:')
        print('/------------------------------------\\')
        print('|  Subset                |  # Images  |')
        print('|-------------------------------------|')
        print('|  Train (Quadrant)      | {:8d}      |'.format(len(quadrant) - num_val_quadrant))
        print('|  Train (Enumeration)   | {:8d}      |'.format(len(quadrant_enum) - num_val_quadrant_enum))
        print('|  Train (Total)         | {:8d}      |'.format(len(merged_data) - num_val_quadrant - num_val_quadrant_enum))
        print('|-------------------------------------|')
        print('|  Val (Quadrant)        | {:8d}      |'.format(num_val_quadrant))
        print('|  Val (Enumeration)     | {:8d}      |'.format(num_val_quadrant_enum))
        print('|  Val (Total)           | {:8d}      |'.format(num_val_quadrant + num_val_quadrant_enum))
        print('|-------------------------------------|')
        print("| Image enhancement: {:5s} |".format("ON" if self.enhance_images else "OFF"))
        print('\\------------------------------------/')

        # Process and save images
        for new_file_name in tqdm(merged_data.keys(), desc="Processing merged YOLO data", unit="images"):
            data_info = merged_data[new_file_name]
            
            # Determine output subdirectory
            output_subdir = 'val' if data_info['is_val'] else 'train'
            display_file_name = new_file_name.replace("train_", "val_") if data_info['is_val'] else new_file_name

            # Prepare label file
            txt_path = os.path.join(self.yolo_output_dir, output_subdir, display_file_name.replace(".png", ".txt"))
            with open(txt_path, 'w') as f:
                f.write("\n".join(data_info['labels']))

            # Copy the image to the YOLO output directory
            # In case the original filename is needed, it can be accessed via data_info['original_filename']
            # In this way we can keep track of the original filename when we deal with quadrant_enum images (since they are renamed)
            # == Before copying, we must open the image and apply some preprocessing, then we can save it
            
            # Get image paths
            image_path = os.path.join(data_info['src_dir'], data_info['original_filename'])
            dest_path = os.path.join(self.yolo_output_dir, output_subdir, display_file_name)
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image {image_path} does not exist.")
            else:
                # Preprocess the image with Sharpening, Contrast Adjustment using Histogram Equalization and Gaussian Filtering
                if self.enhance_images:
                    augmented_image = preprocess_image(image_path)
                    cv2.imwrite(dest_path, augmented_image)
                else:
                    # Copy the image without enhancement
                    shutil.copy(image_path, dest_path)

    def convert_to_yolo_format(self, json_data):
        """
        Converts the dataset to YOLO format for training or validation.

        Args:
            json_data (dict): The JSON data containing annotations and images.
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
            if "category_id_1" in ann and "category_id_2" in ann:
                class_id = ann["category_id_2"] # Tooth class ID offset
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
