import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# Local imports
sys.path.append("./")
from misc.augmentations import preprocess_image

class PeriapicalDatasetDet():
    def __init__(self, dataset_configs):
        # Generic variables
        self.dataset_name = dataset_configs["name"]
        self.data_path = dataset_configs["path"]        
        self.create_yolo_version = dataset_configs["create_yolo_version"] # If True, the dataset will be converted to the YOLO format
        self.enhance_images = dataset_configs["enhance_images"] # If True, the images will be enhanced using the preprocess_image function
                
        self.image_path = os.path.join(self.data_path, 'Augmentation JPG Images')
        self.annotation_path = os.path.join(self.data_path, 'Image Annots')

        self.all_data = os.listdir(self.image_path)
        self.labels = os.listdir(self.annotation_path)

        # Sort the labels and all_data based on the numeric part of the filenames
        self.labels = sorted(self.labels, key=lambda x: int(x.replace('.xml', '')))
        self.all_data = sorted(self.all_data, key=lambda x: int(x.replace('.jpg', '')))

        # Bad labels (They do have annotations but do not have the corresponding images)
        bad_labels = {'18265', '016094', '016134', '16091', '05252', '16093', '31195', '16133', '16131'}
        self.labels = [label for label in self.labels if label.replace('.xml', '') not in bad_labels]
        self.all_data = [data for data in self.all_data if data.replace('.jpg', '') not in bad_labels]
        
        # Class mapping
        self.class_mapping = {
            3: 0, # PAI 3 relabeled to 0
            4: 1, # PAI 4 relabeled to 1
            5: 2, # PAI 5 relabeled to 2
        }

        # Create the YOLO dataset only if the flag is set to True
        if self.create_yolo_version:

            # Output directory for YOLO labels (create if necessary)
            self.yolo_output_dir = os.path.join(self.data_path, "YOLO_dataset")

            if not os.path.exists(os.path.join(self.yolo_output_dir, "train")):
                os.makedirs(os.path.join(self.yolo_output_dir, "train"), exist_ok=True)
            if not os.path.exists(os.path.join(self.yolo_output_dir, "val")):
                os.makedirs(os.path.join(self.yolo_output_dir, "val"), exist_ok=True)
            
            # Convert the dataset to YOLO format
            data = self.convert_to_yolo_format()
            
            # Process the data between Train and Validation sets
            train_split = 0.8
            validation_split = 0.2
            
            # Shuffle the data using random numpy
            np.random.seed(2047315)
            np.random.shuffle(self.all_data)
            np.random.shuffle(self.labels)
            
            self.train_labels = self.labels[0:int(len(self.labels) * train_split)]
            self.validation_labels = self.labels[int(len(self.labels) * train_split):]

            # Plot statistics of the dataset (len of train and val sets)
            print('Image Dataset statistics:')
            print('/-------------------------------------\\')
            print('|  Subset                |  # Images  |')
            print('|-------------------------------------|')
            print('|  Train                 | {:5d}      |'.format(len(self.train_labels)))
            print('|-------------------------------------|')
            print('|  Val                   | {:5d}      |'.format(len(self.validation_labels)))
            print('|-------------------------------------|')
            print("| Image enhancement: {:16s} |".format("ON" if self.enhance_images else "OFF"))
            print('\\-------------------------------------/')

            # Process the data
            self.process_yolo_data(data, self.train_labels, self.validation_labels)
            
            # Create the data.yaml file
            self.create_yaml(self.yolo_output_dir)

        print("Successfully loaded the dataset!")

    def convert_to_yolo_format(self):
        """
        Converts Pascal VOC XML annotations to YOLO format.

        Returns:
            dict: A mapping from image filename to list of YOLO format annotations.
        """
        yolo_labels = defaultdict(list)

        for label_file in tqdm(self.labels, desc="Converting to YOLO format", unit="file"):
            # Parse the XML
            xml_path = os.path.join(self.annotation_path, label_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # The corresponding image file is the same name as the XML file
            filename = label_file.replace('.xml', '.jpg')
            filename_no_ext = filename.split('.')[0]
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            # Extract the relevant information from the XML file
            infos = {'rotated': False, 'mirrored': False,
                     'noise': False, 'scaling': False}

            if filename_no_ext.endswith('2'):
                infos['noise'] = True
            if filename_no_ext.endswith('3'):
                infos['scaling'] = True
            if filename_no_ext.endswith('4'):
                infos['mirrored'] = True
            if filename_no_ext.endswith('5'):
                infos['rotated'] = True
                # In this case we must open the image and get its size
                # That's because images are rotated -90° or -180°, so the only way to get the correct size is to open the image
                height, width, _ = cv2.imread(os.path.join(self.image_path, filename)).shape
                
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_id = self.class_mapping.get(int(class_name), -1)  # Map to new class ID

                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                # Ensure bounding box coordinates are valid
                if xmin > xmax:
                    xmin, xmax = xmax, xmin
                if ymin > ymax:
                    ymin, ymax = ymax, ymin

                # Convert to YOLO format
                x_center = ((xmin + xmax) / 2) / width
                y_center = ((ymin + ymax) / 2) / height
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height

                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
                yolo_labels[filename].append(yolo_line)

        return yolo_labels

    def process_yolo_data(self, data, train_labels, validation_labels):
        """
        Processes the YOLO data and saves it to the appropriate directories.

        Args:
            data (dict): The YOLO formatted data.
            train_labels (list): List of training labels.
            validation_labels (list): List of validation labels.
        """
        for filename, annotations in tqdm(data.items(), desc="Processing YOLO data", unit="file"):
            # Determine if the file is for training or validation
            if filename.replace('.jpg', '.xml') in train_labels:
                output_dir = os.path.join(self.yolo_output_dir, "train")
            elif filename.replace('.jpg', '.xml') in validation_labels:
                output_dir = os.path.join(self.yolo_output_dir, "val")
            else:
                raise ValueError(f"Filename {filename} not found in either train or validation labels.")

            # Save the annotations to a .txt file
            with open(os.path.join(output_dir, filename.replace('.jpg', '.txt')), 'w') as f:
                for annotation in annotations:
                    f.write(annotation + '\n')

            # Get image paths
            image_path = os.path.join(self.image_path, filename)
            dest_path = os.path.join(output_dir, filename)
            
            if not os.path.exists(image_path):
                # Rely on another image path if the first one does not exist
                image_path = image_path.replace('Augmentation JPG Images', 'Original JPG Images')
                if not os.path.exists(image_path):
                    # If the image does not exist in either location, raise an error
                    raise FileNotFoundError(f"Image {image_path} does not exist.")
                
            # Preprocess the image with Sharpening, Contrast Adjustment using Histogram Equalization and Gaussian Filtering
            if self.enhance_images:
                augmented_image = preprocess_image(image_path)
                cv2.imwrite(dest_path, augmented_image)
            else:
                # Copy the image without enhancement
                shutil.copy(image_path, dest_path)
                
    def create_yaml(self, output_dir):
        """
        Creates a data.yaml file for YOLOv5.

        Args:
            output_dir (str): The directory where the data.yaml file will be saved.
        """
        yaml_content = {
            "train": os.path.join(os.getcwd(), output_dir, "train"),
            "val": os.path.join(os.getcwd(), output_dir, "val"),
            "nc": 3,
            "names": {
                0: "PAI 3",
                1: "PAI 4",
                2: "PAI 5"
            }
        }

        with open(os.path.join(output_dir, "data.yaml"), 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=None, sort_keys=False)

class PeriapicalDataset():
    def __init__(self, data_path):
        # Generic variables
        self.dataset_name = 'periapical'
        self.data_path = os.path.join(data_path, 'Periapical Dataset', 'Periapical Lesions')
        
        self.image_path = os.path.join(self.data_path, 'Augmentation JPG Images')
        self.annotation_path = os.path.join(self.data_path, 'Image Annots')
        
        self.all_data = os.listdir(self.image_path)
        self.labels = os.listdir(self.annotation_path)

        # Sort the labels and all_data based on the numeric part of the filenames
        self.labels = sorted(self.labels, key=lambda x: int(re.search(r'\d+', x).group()))
        self.all_data = sorted(self.all_data, key=lambda x: int(re.search(r'\d+', x).group()))

        # Split the dataset into train and validation sets
        train_split = 0.8
        validation_split = 0.2
        
        # Shuffle the data using random numpy
        np.random.seed(2047315)
        np.random.shuffle(self.all_data)
        np.random.shuffle(self.labels)
        
        self.train_labels = self.labels[0:int(len(self.labels) * train_split)]
        self.validation_labels = self.labels[int(len(self.labels) * train_split):]
        
        # Train
        self.train = self.get_list(self.train_labels, self.all_data)
        
        # Validation
        self.validation = self.get_list(self.validation_labels, self.all_data)
        
        # Define the mapping of old class IDs to new class IDs
        self.class_mapping = {
            '3': 0, # PAI 3 relabeled to 0
            '4': 1, # PAI 4 relabeled to 1
            '5': 2, # PAI 5 relabeled to 2
        }

        # Re-label the IDs in the dataset
        self.train = self.relabel_ids(self.train)
        self.validation = self.relabel_ids(self.validation)

        print("Successfully loaded the dataset!")
        
    # Extract infos for each image
    def get_labels(self, file_path):
        # Load and parse the XML file
        tree = ET.parse(file_path)  # replace with the actual path
        root = tree.getroot()

        # Accessing elements
        filename = root.find('filename').text
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        depth = int(root.find('size/depth').text)

        # Initialize variables
        item_info = {
            'filename': filename,
            'width': width,
            'height': height,
            'depth': depth,
            'objects': []
        }

        # Accessing object info
        for obj in root.findall('object'):
            obj_info = {
                'name': obj.find('name').text,
                'xmin': int(obj.find('bndbox/xmin').text),
                'ymin': int(obj.find('bndbox/ymin').text),
                'xmax': int(obj.find('bndbox/xmax').text),
                'ymax': int(obj.find('bndbox/ymax').text)
            }
            item_info['objects'].append(obj_info)
        
        return item_info

    # Get the list of images and their corresponding labels
    def get_list(self, labels, all_images):
        dataset = []
        
        # Filter the all_images list to only include images that are in the labels list
        labels = [label.replace('.xml', '') for label in labels]
        images = [os.path.join(self.image_path, img) for img in all_images if img.replace('.jpg', '') in labels]
        
        for img_path in tqdm(images, desc="Loading dataset", unit="image"):
            # open the label corresponding to the image
            img_name = os.path.basename(img_path).split('.')[0]
            label_path = os.path.join(self.annotation_path, img_name + '.xml')
            labels = self.get_labels(label_path)
            
            # Extract the relevant information from the XML file
            infos = {'rotated': False, 'mirrored': False,
                     'noise': False, 'scaling': False}

            if img_name.endswith('2'):
                infos['noise'] = True
            if img_name.endswith('3'):
                infos['scaling'] = True
            if img_name.endswith('4'):
                infos['mirrored'] = True
            if img_name.endswith('5'):
                infos['rotated'] = True

            details = (img_path, labels['width'], labels['height'], labels['depth'])
            
            objs = []
            for obj in labels['objects']:
                # Extract the relevant information from the XML file
                periapical_class = obj['name']
                xmin = obj['xmin']
                ymin = obj['ymin']
                xmax = obj['xmax']
                ymax = obj['ymax']
                
                # Append the details to the dataset
                objs.append((periapical_class, xmin, ymin, xmax, ymax))

            dataset.append(details + (objs,) + (infos,))

        return dataset

    def relabel_ids(self, dataset):      
        # Create a new dataset with relabeled IDs
        relabeled_dataset = []

        for data in dataset:
            img_path, width, height, depth, objects, infos = data
            
            objs = []
            for obj in objects:
                periapical_class, xmin, ymin, xmax, ymax = obj
                
                # Convert the class name to an integer ID
                periapical_class = self.class_mapping.get(periapical_class, -1)
                objs.append((periapical_class, xmin, ymin, xmax, ymax))

            relabeled_dataset.append((img_path, width, height, depth, objs, infos))

        return relabeled_dataset
