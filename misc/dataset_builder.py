import copy
import os
import pickle
import random
import sys
from collections import defaultdict
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets import periapical
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

sys.path.append("./")
from misc.augmentations import Transformations
from misc.utils import read_image

class DatasetBuilder():
    def __init__(self, data_path, dataset_name, augmentation_configs=None):
        """
        Args:
            data_path (string): Path to the data file.
            dataset_name (string / list): Name of the dataset (periapical).
            augmentation_configs (dict): Dictionary containing the augmentation configurations.
        """
        self.data_path = data_path
        self.dataset_name = dataset_name

        if self.dataset_name == 'periapical':
            self.dataset = periapical.PeriapicalDataset(self.data_path)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # Transformations for the dataset
        self.transforms = Transformations(configs=augmentation_configs)

        # Create the train and validation datasets
        self.train_set = ImageDataset(dataset_name=self.dataset.dataset_name,
                                      data=self.dataset.train,
                                      transform=self.transforms.get_train_transform(),
                                      train=True)
        
        # If the query or gallery is empty, then the validation set is None
        if len(self.dataset.validation) == 0:
            self.validation_set = None
        else:
            self.validation_set = ImageDataset(dataset_name=self.dataset.dataset_name,
                                                data=self.dataset.validation,
                                                transform=self.transforms.get_val_transform(),
                                                train=False)
        verbose = True
        if verbose:
            print('Image Dataset statistics:')
            print('/--------------------------\\')
            print('|  Subset       | # Images |')
            print('|--------------------------|')
            print('|  Train        | {:8d} |'.format(len(self.dataset.train)))
            print('|  Validation   | {:8d} |'.format(len(self.dataset.validation)))
            print('\\--------------------------/')

class ImageDataset(Dataset):
    def __init__(self,
                 dataset_name: str,
                 data,
                 transform=None,
                 train=True):
        """
        Args:
            dataset_name (string): Name of the dataset (periapical).
            data (list): List of tuples containing image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
            train (bool): If True, the dataset is used for training. If False, it is used for validation.
        """
        self.dataset_name = dataset_name
        self.data = data
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[str, torch.Tensor, int, int, int, int, int, str]:
        img_path, width, height, depth, objects, infos = self.data[idx]

        # Read image in PIL format
        img = read_image(img_path)
        real_width, real_height = img.size

        rotated = infos['rotated']
        mirrored = infos['mirrored']

        imgs = []
        
        final_objects = objects.copy() # Create a deep copy of the objects list
        
        # === When we read the image, we need to extract patches of the bounding boxes ===
        for i, obj in enumerate(objects):
            # Convert potentially modified coordinates to integers for cropping
            # Ensure coordinates are within image bounds after transformation
            x1 = max(0, int(obj[1]))
            y1 = max(0, int(obj[2]))
            x2 = min(real_width, int(obj[3]))
            y2 = min(real_height, int(obj[4]))

            # Ensure coordinates are valid (x1 < x2, y1 < y2)
            if x1 >= x2 or y1 >= y2:
                #print(f"Invalid bounding box coordinates: ({x1}, {y1}, {x2}, {y2}) for image {img_path}")
                # We must remove the invalid bounding box from the list of objects
                final_objects.remove(obj)

                # # Fix for rotated and mirrored images with invalid bounding boxes (doesn't work with for example: 02255.jpg)
                # if x1 >= x2:
                #     x2 = x1
                #     x1 = x2
                # if y1 >= y2:
                #     y2 = y1
                #     y1 = y2
            else:
                # Extract the bounding box from the image
                img_cropped = img.crop((x1, y1, x2, y2))

                # Apply the transform if it exists
                if self.transform is not None:
                    img_cropped = self.transform(img_cropped)
                    
                imgs.append(img_cropped)

        return img_path, imgs, width, height, depth, final_objects, infos

    def train_collate_fn(self, batch):# -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, str]:
        img_paths, imgs, widths, heights, depths, objects, infos = zip(*batch)

        # Flatten the nested list of tensors
        flattened_imgs = [img for sublist in imgs for img in sublist]

        # Stack the tensors
        imgs = torch.stack(flattened_imgs, dim=0)

        # Convert the labels to a tensor
        for objs in objects:
            for i, obj in enumerate(objs):
                # Convert the tuple to a list to allow modification
                obj_list = list(obj)
                obj_list[0] = torch.tensor(obj_list[0], dtype=torch.int64)
                # Convert it back to a tuple and update the list
                objs[i] = tuple(obj_list)
        
        if len(imgs) != len([obj for sublist in objects for obj in sublist]):
            raise ValueError(f"Number of images ({len(imgs)}) does not match number of objects ({len(objects)})")
        
        return img_paths, imgs, widths, heights, depths, objects, infos

    def val_collate_fn(self, batch):# -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, str]:
        img_paths, imgs, widths, heights, depths, objects, infos = zip(*batch)

        # Flatten the nested list of tensors
        flattened_imgs = [img for sublist in imgs for img in sublist]

        # Stack the tensors
        imgs = torch.stack(flattened_imgs, dim=0)

        # Convert the labels to a tensor
        for objs in objects:
            for i, obj in enumerate(objs):
                # Convert the tuple to a list to allow modification
                obj_list = list(obj)
                obj_list[0] = torch.tensor(obj_list[0], dtype=torch.int64)
                # Convert it back to a tuple and update the list
                objs[i] = tuple(obj_list)

        return img_paths, imgs, widths, heights, depths, objects, infos
