import json
import os
import random
import shutil
import sys
from zipfile import ZipFile

import cv2
import yaml
from tqdm import tqdm
import subprocess

# Local imports
sys.path.append("./")
from misc.augmentations import preprocess_image

# Define the URL and the target path
download_path = os.path.expanduser("./DualLabel.zip")

# https://www.kaggle.com/datasets/zwbzwb12341234/a-dual-labeled-dataset
class DualLabel():
    def __init__(self, dataset_configs):
        # Generic variables
        self.dataset_name = dataset_configs["name"]
        self.data_path = dataset_configs["path"]
        self.create_yolo_version = dataset_configs["create_yolo_version"] # If True, the dataset will be converted to the YOLO format
        self.enhance_images = dataset_configs["enhance_images"] # If True, the images will be enhanced using the augmentations defined in misc/augmentations.py

        # Directories
        self.ann_folder = os.path.join(self.data_path, "labels")
        self.img_folder = [os.path.join(self.data_path, f'images{i}') for i in range (1, 5)]

        # Create a mapping for storing in which directory of img_folder the images are located
        self.img_mapping = {}
        for folder in self.img_folder:
            for img in os.listdir(folder):
                if img.endswith(".png"):
                    self.img_mapping[img] = folder

        # Create a mapping for the classes
        self.classes = {}
        for ann_file in os.listdir(self.ann_folder):
            # Load the annotation file
            with open(os.path.join(self.ann_folder, ann_file), 'r') as f:
                ann_data = json.load(f)

            # Iterate through the annotation objects
            for ann in ann_data["shapes"]:
                # Get the class label
                cls = ann["label"]

                # Add the class to the dictionary if not already present
                if cls not in self.classes:
                    self.classes[cls] = len(self.classes)  # Assign an incremental index
                    
        # Re-index the classes to start from 0 in ascending order
        self.classes = {k: i for i, (k, _) in enumerate(sorted(self.classes.items(), key=lambda x: int(x[0])))}
            
        # Create the YOLO dataset only if the flag is set to True
        if self.create_yolo_version:

            # Output directory for YOLO labels (create if necessary)
            self.yolo_output_dir = os.path.join(self.data_path, "YOLO_dataset")

            if not os.path.exists(os.path.join(self.yolo_output_dir, "train")):
                os.makedirs(os.path.join(self.yolo_output_dir, "train"), exist_ok=True)
            if not os.path.exists(os.path.join(self.yolo_output_dir, "val")):
                os.makedirs(os.path.join(self.yolo_output_dir, "val"), exist_ok=True)

            # We need to simply convert the classes into a YOLO format and then proceed with the splits
            yolo_data, yolo_data_background = self.convert_to_yolo_format()

            # Process the final data
            self.create_splits(yolo_data, yolo_data_background, self.yolo_output_dir, splits=(0.8, 0.2))

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
            "names": names_dict,
        }

        with open(os.path.join(yolo_dir, "data.yaml"), 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=None, sort_keys=False)

    def download_and_extract(self, download_path, extract_path):
        try:
            # Download the file with progress bar
            print("Downloading dataset...")
            
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path, exist_ok=True)

            curl_command = [
                "curl",
                "https://www.kaggle.com/datasets/zwbzwb12341234/a-dual-labeled-dataset/download?datasetVersionNumber=5",
                "-H", "accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "-H", "accept-language: it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6,zh;q=0.5",
                "-b", "ka_sessionid=0813a8561a80f7325e19229155c256f7; GCLB=CJntudSay6Kd2gEQAw; __Host-KAGGLEID=CfDJ8KT8tnOr7fFFm_byYmusL7iaDDlNkpOcp9eK1AHD5gagQCZzcnKD0jiviwi3H0Z9NGKENM5nJjJqgpjOsBVD2m6n8YYmGtk2gRHL3yHC2rlE2moz8UaBmc4K; build-hash=3988b190d3d98f2fa4478d6c003bcb2e08c3f2a5; CSRF-TOKEN=CfDJ8KT8tnOr7fFFm_byYmusL7hAVQGUCGfjLjl3ykcHSStdQYsil1vgO1AKFbZpmf6qmQw3WrU_hxZH8J1UsZoZMw4XyeoGQUy8WFeUdf8pBw; XSRF-TOKEN=CfDJ8KT8tnOr7fFFm_byYmusL7i41pLNm59lWjdV1mpL9oWdt9mMU52PX4YZ4eRHuzmlvKLQk1WS9TMzdp91mwLV2MXihXxHvwj64Yar4PHVvYrV7UBAqUTXUYTjv2eDJqeCsk8ZIg9OhUY3L66zJSKODpE; CLIENT-TOKEN=eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpc3MiOiJrYWdnbGUiLCJhdWQiOiJjbGllbnQiLCJzdWIiOiJnaWFubWFyY29zY2FyYW5vIiwibmJ0IjoiMjAyNS0wNC0xMVQxNzowMTowOC41MTIzMjU0WiIsImlhdCI6IjIwMjUtMDQtMTFUMTc6MDE6MDguNTEyMzI1NFoiLCJqdGkiOiJlMGUzYjc0NC1iYjk5LTQzODAtYjViZC1lNTFiZWQxNmYxNTUiLCJleHAiOiIyMDI1LTA1LTExVDE3OjAxOjA4LjUxMjMyNTRaIiwidWlkIjoxMjg0NDUzMCwiZGlzcGxheU5hbWUiOiJHaWFubWFyY28gU2NhcmFubyIsImVtYWlsIjoiZ2lhbm1hcmNvc2NhcmFub0BnbWFpbC5jb20iLCJ0aWVyIjoibm92aWNlIiwidmVyaWZpZWQiOmZhbHNlLCJwcm9maWxlVXJsIjoiL2dpYW5tYXJjb3NjYXJhbm8iLCJ0aHVtYm5haWxVcmwiOiJodHRwczovL3N0b3JhZ2UuZ29vZ2xlYXBpcy5jb20va2FnZ2xlLWF2YXRhcnMvdGh1bWJuYWlscy8xMjg0NDUzMC1nci5qcGciLCJmZiI6WyJCYXRjaEltcG9ydEtlcm5lbHNGcm9tQ29sYWIiLCJDb21wZXRpdGlvblNpbXVsYXRpb25TZXR0aW5ncyIsIkNvcHlNb2RlbEluc3RhbmNlVmVyc2lvbiIsIkxpbmtPcGVuTWxEYXRhc2V0cyIsIkRhdGFzZXRWZXJzaW9uc0luRmxpZ2h0IiwiS2VybmVsc09wZW5JbkNvbGFiTG9jYWxVcmwiLCJNZXRhc3RvcmVDaGVja0FnZ3JlZ2F0ZUZpbGVIYXNoZXMiLCJCYWRnZXMiLCJDaGVja0VmZmljaWVudEV4dGVuc2lvbnMiLCJVc2VyTGljZW5zZUFncmVlbWVudFN0YWxlbmVzc1RyYWNraW5nIiwiQWRtaW5Pbmx5T3JnYW5pemF0aW9uQ3JlYXRpb24iLCJOZXdPcmdhbml6YXRpb25SZXF1ZXN0Rm9ybSIsIkdyb3VwcyIsIkdyb3Vwc0ludGVncmF0aW9uIiwiRW5hYmxlU3BvdGxpZ2h0Q29tbXVuaXR5Q29tcGV0aXRpb25zU2hlbGYiLCJGYWN0QmVuY2hMZWFkZXJib2FyZCIsIkRhdGFzZXRzQ29kZVBvcFVwIiwiS2VybmVsc1ByaXZhdGVQYWNrYWdlTWFuYWdlciIsIkxvY2F0aW9uU2hhcmluZ09wdE91dCIsIkRhdGFzZXRzUGFycXVldFN1cHBvcnQiLCJHZW1tYUxpY2Vuc2VSZXZpc2lvbjMiLCJLZXJuZWxzRmlyZWJhc2VMb25nUG9sbGluZyIsIktlcm5lbHNEcmFmdFVwbG9hZEJsb2IiLCJLZXJuZWxzU2F2ZUNlbGxPdXRwdXQiLCJGcm9udGVuZEVycm9yUmVwb3J0aW5nIiwiQWxsb3dGb3J1bUF0dGFjaG1lbnRzIiwiVGVybXNPZlNlcnZpY2VCYW5uZXIiLCJEYXRhc2V0VXBsb2FkZXJEdXBsaWNhdGVEZXRlY3Rpb24iXSwiZmZkIjp7Ik1vZGVsSWRzQWxsb3dJbmZlcmVuY2UiOiIiLCJNb2RlbEluZmVyZW5jZVBhcmFtZXRlcnMiOiJ7IFwibWF4X3Rva2Vuc1wiOiAxMjgsIFwidGVtcGVyYXR1cmVcIjogMC40LCBcInRvcF9rXCI6IDUgfSIsIkdldHRpbmdTdGFydGVkQ29tcGV0aXRpb25zIjoiMzEzNiwxMDIxMSw1NDA3LDM0Mzc3IiwiU3BvdGxpZ2h0Q29tbXVuaXR5Q29tcGV0aXRpb24iOiI5MTE5Niw5MTQ1MSw5MTQ0OCw4OTg1MCw4ODYxMiw5NDY4OSw5NzU2OSIsIlN0c01pbkZpbGVzIjoiNzUwMDAiLCJTdHNNaW5HYiI6IjEiLCJDbGllbnRScGNSYXRlTGltaXRRcHMiOiI0MCIsIkNsaWVudFJwY1JhdGVMaW1pdFFwbSI6IjUwMCIsIkFkZEZlYXR1cmVGbGFnc1RvUGFnZUxvYWRUYWciOiJkaXNhYmxlZCIsIktlcm5lbEVkaXRvckF1dG9zYXZlVGhyb3R0bGVNcyI6IjMwMDAwIiwiS2VybmVsc0w0R3B1Q29tcHMiOiI4NjAyMyw4NDc5NSw4ODkyNSw5MTQ5NiIsIkZlYXR1cmVkQ29tbXVuaXR5Q29tcGV0aXRpb25zIjoiNjAwOTUsNTQwMDAsNTcxNjMsODA4NzQsODE3ODYsODE3MDQsODI2MTEsODUyMTAiLCJFbWVyZ2VuY3lBbGVydEJhbm5lciI6Int9IiwiQ29tcGV0aXRpb25NZXRyaWNUaW1lb3V0TWludXRlcyI6IjMwIiwiS2VybmVsc1BheVRvU2NhbGVQcm9QbHVzR3B1SG91cnMiOiIzMCIsIktlcm5lbHNQYXlUb1NjYWxlUHJvR3B1SG91cnMiOiIxNSIsIkRhdGFzZXRzU2VuZFBlbmRpbmdTdWdnZXN0aW9uc1JlbWluZGVyc0JhdGNoU2l6ZSI6IjEwMCJ9LCJwaWQiOiJrYWdnbGUtMTYxNjA3Iiwic3ZjIjoid2ViLWZlIiwic2RhayI6IkFJemFTeUE0ZU5xVWRSUnNrSnNDWldWei1xTDY1NVhhNUpFTXJlRSIsImJsZCI6IjM5ODhiMTkwZDNkOThmMmZhNDQ3OGQ2YzAwM2JjYjJlMDhjM2YyYTUifQ.",
                "-H", "priority: u=0, i",
                "-H", "referer: https://www.kaggle.com/datasets/zwbzwb12341234/a-dual-labeled-dataset",
                "-H", 'sec-ch-ua: "Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"' \
                "-H", 'sec-ch-ua-mobile: ?0' \
                "-H", 'sec-ch-ua-platform: "Windows"' \
                "-H", 'sec-fetch-dest: document' \
                "-H", 'sec-fetch-mode: navigate' \
                "-H", 'sec-fetch-site: same-origin' \
                "-H", 'sec-fetch-user: ?1' \
                "-H", 'upgrade-insecure-requests: 1' \
                "-H", 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36'
            ]

            try:
                subprocess.run(curl_command, check=True)
            except subprocess.CalledProcessError as e:
                print("Download failed:", e)

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

    def convert_to_yolo_format(self):       
        # Get the JSON files
        ann_files = [f for f in os.listdir(self.ann_folder) if f.endswith(".json")]
        img_files = [f for folder in self.img_folder for f in os.listdir(folder) if f.endswith(".png")]
        
        # Sort the files
        ann_files.sort(key=lambda x: str(x.split('.')[0]))
        img_files.sort(key=lambda x: str(x.split('.')[0]))

        # Check if the number of JSON files and images match
        assert len(ann_files) == len(img_files), "Number of JSON files and images do not match."

        # Iterate through the JSON files and create the YOLO format
        yolo_data = {}
        yolo_data_background = {}
        
        for ann_file in tqdm(ann_files, desc="Processing images", unit="file"):               
            # Load the annotation file
            with open(os.path.join(self.ann_folder, ann_file), 'r') as f:
                ann_data = json.load(f)

            # Get the image file name
            img_file = ann_data["imagePath"]
            
            if img_file not in yolo_data:
                yolo_data[img_file] = []
            else: # In this case, the image is already present in the yolo_data dictionary, so we might consider this as a background image
                yolo_data_background[img_file] = []
                continue
                
            # Get size
            img_h = ann_data["imageHeight"]
            img_w = ann_data["imageWidth"]

            # Iterate through the annotation objects
            for ann in ann_data["shapes"]:

                # Get the class and convert it to a 0-based index
                cls = ann["label"]
                
                # Get class from the mapping
                if cls not in self.classes:
                    raise ValueError(f"Class '{cls}' not found in the mapping.")
                else:
                    cls = self.classes[cls]

                # Get shape type
                type = ann["shape_type"]

                # Extract the bounding box coordinates
                if type == "polygon" or type == "linestrip":
                    # For polygons, we'll include all the points
                    points = ann["points"]
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
                    raise ValueError(f"Unsupported shape type: {type}")
            
            # Before continuing with the loop, we can sort the yolo_data[img_file] list to ensure that the classes are in ascending order
            yolo_data[img_file].sort(key=lambda x: int(x.split()[0]))
            
        return yolo_data, yolo_data_background

    def create_splits(self, yolo_data: dict, yolo_data_background: dict, output_dir, splits=(0.8, 0.2)):
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
                with open(os.path.join(split_dir, img.replace(".png", ".txt")), 'w') as f:
                    f.write("\n".join(yolo_data[img]))
                    
                # Copy the image file
                # == Before copying, we need to preprocess the image
                
                # Get image paths from the mapping
                folder_path = self.img_mapping.get(img, None)
                if folder_path is None:
                    raise FileNotFoundError(f"Image {img} not found in the mapping.")
                
                image_path = os.path.join(folder_path, img)
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

        # Copy background images to the train split
        for img in yolo_data_background.keys():
            # Get image paths from the mapping
            folder_path = self.img_mapping.get(img, None)

            if folder_path is None:
                raise FileNotFoundError(f"Background image {img} not found in the mapping.")

            image_path = os.path.join(folder_path, img)
            dest_path = os.path.join(output_dir, "train", img)

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image {image_path} does not exist.")
            else:
                shutil.copy(image_path, dest_path)
                
                # Also remove any possible labeling file created for the background images
                label_path = os.path.join(output_dir, "train", img.replace(".png", ".txt"))
                if os.path.exists(label_path):
                    os.remove(label_path)
