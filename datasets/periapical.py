import glob
import os
import pickle
import random
import re
import shutil
import xml.etree.ElementTree as ET
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

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
