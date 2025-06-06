import json
import os
import random
import shutil
import sys
from zipfile import ZipFile

import cv2
import requests
import yaml
from PIL import Image
from tqdm import tqdm

# Local imports
sys.path.append("./")
from misc.augmentations import preprocess_image

# Define the URL and the target path
download_path = os.path.expanduser("./archive.zip")

# https://datasetninja.com/teeth-segmentation#download
class TeethSeg():
    def __init__(self, dataset_configs):
        # Generic variables
        self.dataset_name = dataset_configs["name"]
        self.data_path = dataset_configs["path"]
        self.url = "https://www.kaggle.com/api/v1/datasets/download/humansintheloop/teeth-segmentation-on-dental-x-ray-images" # URL to download the dataset
        self.create_yolo_version = dataset_configs["create_yolo_version"] # If True, the dataset will be converted to the YOLO format
        self.enhance_images = dataset_configs["enhance_images"] # If True, the images will be enhanced using the augmentations defined in misc/augmentations.py

        # Download the Dataset if it doesn't exist
        if not os.path.exists(self.data_path):
            self.download_and_extract(self.url, download_path, self.data_path)

        # Directories
        self.ann_folder = os.path.join(self.data_path, "Teeth Segmentation JSON", "d2", "ann")
        self.img_folder = os.path.join(self.data_path, "Teeth Segmentation JSON", "d2", "img")

        # Get the class map
        # From source model load classes that this dataset contains
        self.meta = json.load(open(os.path.join(self.data_path, "Teeth Segmentation JSON", "meta.json")))
        self.obj_class_to_machine_color = json.load(open(os.path.join(self.data_path, "Teeth Segmentation JSON", "obj_class_to_machine_color.json")))
        
        self.classes = {}
        for (index, entry) in enumerate(self.meta["classes"]):
            # Get the title
            title = entry["title"]
            # Get the color
            color = self.obj_class_to_machine_color[title]
            # Add the class to the dictionary
            self.classes[title] = (index, color)

        # Create the YOLO dataset only if the flag is set to True
        if self.create_yolo_version:

            # Output directory for YOLO labels (create if necessary)
            self.yolo_output_dir = os.path.join(self.data_path, "YOLO_dataset")

            if not os.path.exists(os.path.join(self.yolo_output_dir, "train")):
                os.makedirs(os.path.join(self.yolo_output_dir, "train"), exist_ok=True)
            if not os.path.exists(os.path.join(self.yolo_output_dir, "val")):
                os.makedirs(os.path.join(self.yolo_output_dir, "val"), exist_ok=True)

            # We need to simply convert the classes into a YOLO format and then proceed with the splits
            yolo_data = self.convert_to_yolo_format(self.ann_folder, self.img_folder)
                        
            # Process the final data
            self.create_splits(yolo_data, self.yolo_output_dir, splits=(0.8, 0.2))

            # Create the data.yaml file
            self.create_yaml(self.yolo_output_dir, self.classes)

            print("YOLO dataset and data.yaml successfully generated!")

    def create_yaml(self, yolo_dir, class_map):
        # Create a dictionary with class names as keys and class IDs as values
        names_dict = {i: name for i, name in enumerate(class_map.keys())}
    
        yaml_content = {
            "train": os.path.join(os.getcwd(), self.yolo_output_dir, "train"),
            "val": os.path.join(os.getcwd(), self.yolo_output_dir, "val"),
            "nc": len(class_map),
            "names": names_dict
        }
        
        with open(os.path.join(yolo_dir, "data.yaml"), 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=None, sort_keys=False)

    def download_and_extract(self, url, download_path, extract_path):
        try:
            # Download the file with progress bar
            print("Downloading dataset...")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check for HTTP request errors
            
            # Get the total file size from headers
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192  # 8KB chunks
            
            with open(download_path, "wb") as file:
                with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as progress_bar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            print("Download complete.")

            # Extract the zip file
            print("Extracting files...")
            with ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"Files extracted to {extract_path}")

        finally:
            # Remove the ZIP file
            if os.path.exists(download_path):
                os.remove(download_path)
                print("ZIP file deleted.")
            else:
                print("ZIP file not found.")

    def get_polygon_bbox(self, points):
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        return min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)

    def convert_to_yolo_format(self, ann_folder, img_folder):
        # Get the JSON files
        ann_files = [f for f in os.listdir(ann_folder) if f.endswith(".json")]
        img_files = [f for f in os.listdir(img_folder) if f.endswith(".jpg")]
        
        # Sort the files
        ann_files.sort(key=lambda x: int(x.split('.')[0]))
        img_files.sort(key=lambda x: int(x.split('.')[0]))

        # Check if the number of JSON files and images match
        assert len(ann_files) == len(img_files), "Number of JSON files and images do not match."

        # Iterate through the JSON files and create the YOLO format
        yolo_data = {}
        for ann_file in tqdm(ann_files, desc="Processing images", unit="file"):
            # Get the image file name (without extension)
            img_file = ann_file.split(".json")[0]
            
            if img_file not in yolo_data:
                yolo_data[img_file] = []
                
            # Load the annotation file
            with open(os.path.join(ann_folder, ann_file), 'r') as f:
                ann_data = json.load(f)
                        
            # Get size
            img_h = ann_data["size"]["height"]
            img_w = ann_data["size"]["width"]
                        
            # Iterate through the annotation objects
            for ann in ann_data["objects"]:
               
                # Get the class and the bounding box
                cls, _ = self.classes[ann["classTitle"]]
                type = ann["geometryType"]
                
                # Extract the bounding box coordinates
                if type == "rect":
                    # For rectangles, we'll just use the bounding box coordinates
                    x, y, w, h = ann["bbox"]
                    # Convert to YOLO bbox format (center x, center y, width, height)
                    x_center = (x + w / 2) / img_w
                    y_center = (y + h / 2) / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h
                    # Format: class x_center y_center width height
                    yolo_data[img_file].append(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                elif type == "polygon":
                    # segmentation = ann["points"]["exterior"]
                    # x, y, w, h = self.get_polygon_bbox(segmentation)
                    # For polygons, we'll include all the points
                    points = ann["points"]["exterior"]
                    # Normalize all points
                    normalized_points = []
                    for x, y in points:
                        x_norm = x / img_w
                        y_norm = y / img_h
                        normalized_points.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])
                        
                    # Format: class x1 y1 x2 y2 ... xn yn
                    yolo_line = f"{cls} {' '.join(normalized_points)}"
                    yolo_data[img_file].append(yolo_line)
                else:
                    continue

        return yolo_data

    def create_splits(self, yolo_data: dict, output_dir, splits=(0.8, 0.2)):
        # Shuffle the data of yolo_data (which is a list of key-value pairs)
        images = list(yolo_data.keys())
        random.shuffle(images)

        # Calculate split indices
        total_images = len(images)
        train_count = round(splits[0] * total_images)
        val_count = round(splits[1] * total_images)
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]

        print("Image Dataset statistics:")
        print("/-------------------------------\\")
        print("| Subset           | # Images  |")
        print("|-------------------------------|")
        print("| Train            | {:9d} |".format(len(train_images)))
        print("| Val              | {:9d} |".format(len(val_images)))
        print("| Enhancement      | {:>9s} |".format("ON" if self.enhance_images else "OFF"))
        print("\\-------------------------------/")

        # Create the splits
        for split, split_images in zip(["train", "val"], [train_images, val_images]):
            split_dir = os.path.join(output_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_images:
                with open(os.path.join(split_dir, img.replace(".jpg", ".txt")), 'w') as f:
                    f.write("\n".join(yolo_data[img]))
                    
                # Copy the image file
                # == Before copying, we need to preprocess the image
                
                # Get image paths
                image_path = os.path.join(self.img_folder, img)
                dest_path = os.path.join(split_dir, img)
                
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
