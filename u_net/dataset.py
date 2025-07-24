import json
import os

import albumentations as A
import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

class SegmentationDataset(Dataset):
    def __init__(self, data_path, img_size=512, val_size=0.2, random_state=2047315):
        self.data_path = data_path
        self.img_size = img_size
        self.debug = False # Set to True for debugging purposes
        
        # Get all image files from multiple directories
        self.image_files = []
        self.image_paths = []
        
        # Check all image directories (images1, images2, images3, images4)
        for i in range(1, 5):
            img_dir = os.path.join(data_path, f'images{i}')
            if os.path.exists(img_dir):
                for f in os.listdir(img_dir):
                    if f.endswith(('.jpg', '.png', '.jpeg')):
                        self.image_files.append(f)
                        self.image_paths.append(os.path.join(img_dir, f))
        
        self.labels_dir = os.path.join(data_path, 'labels')
        
        # Limit dataset to 100 images for testing if debug is True
        if self.debug:
            self.image_files = self.image_files[:100]
            self.image_paths = self.image_paths[:100]
            print("Debug mode: Limiting dataset to 100 images")
        
        # Create train/val split ONCE
        if len(self.image_files) > 0:
            train_indices, val_indices = train_test_split(
                range(len(self.image_files)),
                test_size=val_size, 
                random_state=random_state
            )
            self.train_indices = train_indices
            self.val_indices = val_indices
        else:
            self.train_indices = []
            self.val_indices = []
        
        print(f"Total dataset: {len(self.image_files)} images")
        print(f"Train indices: {len(self.train_indices)} images")
        print(f"Val indices: {len(self.val_indices)} images")
    
    def get_train_dataset(self, augment=True):
        """Get training subset with augmentation"""
        train_subset = Subset(self, self.train_indices)
        train_subset.dataset = SegmentationDatasetSplit(self, augment=augment)
        return train_subset

    def get_val_dataset(self, augment=False):
        """Get validation subset without augmentation"""
        val_subset = Subset(self, self.val_indices)
        val_subset.dataset = SegmentationDatasetSplit(self, augment=augment)
        return val_subset
    
    def __len__(self):
        return len(self.image_files)
    
    def json_to_mask(self, json_path, img_height, img_width):
        """Convert JSON annotations to segmentation mask"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Create empty mask
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        # Map FDI tooth numbers to class IDs (0-32)
        tooth_to_class = {
            # Upper teeth (1x)
            11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5, 17: 6, 18: 7,
            21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
            # Lower teeth (3x, 4x)
            31: 16, 32: 17, 33: 18, 34: 19, 35: 20, 36: 21, 37: 22, 38: 23,
            41: 24, 42: 25, 43: 26, 44: 27, 45: 28, 46: 29, 47: 30, 48: 31,
            # Supernumerary tooth
            91: 32
        }
        
        for shape in data['shapes']:
            if shape['shape_type'] == 'polygon':
                # Get tooth number and convert to class ID
                tooth_num = int(shape['label'])
                class_id = tooth_to_class.get(tooth_num, -1)
                
                if class_id == -1:
                    raise ValueError(f"Unknown tooth number {tooth_num} in annotation {json_path}")
                
                # Get polygon points
                points = np.array(shape['points'], dtype=np.int32)
                
                # Fill polygon in mask
                cv2.fillPoly(mask, [points], class_id)
        
        return mask
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get original image dimensions
        img_height, img_width = image.shape[:2]
        
        # Load corresponding JSON annotation
        img_name = self.image_files[idx]
        json_name = img_name.replace('.png', '.json').replace('.jpg', '.json').replace('.jpeg', '.json')
        json_path = os.path.join(self.labels_dir, json_name)
        
        if os.path.exists(json_path):
            mask = self.json_to_mask(json_path, img_height, img_width)
        else:
            # Create empty mask if no annotation exists
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            print(f"Warning: No annotation found for {img_name}")
        
        return image, mask

class SegmentationDatasetSplit(Dataset):
    """Wrapper class to apply different transforms to the same base dataset"""
    def __init__(self, base_dataset, augment=True):
        self.base_dataset = base_dataset
        self.augment = augment
        self.transform = None
        
        # Define augmentations
        if self.augment:
            self.transform = A.Compose([
                A.Resize(base_dataset.img_size, base_dataset.img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.pytorch.ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(base_dataset.img_size, base_dataset.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.pytorch.ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]
        
        # Apply transforms
        if self.transform:
            # For Albumentations
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask.long()

# Test the dataset
if __name__ == "__main__":
    # Test dataset loading
    data_path = os.path.join(os.getcwd(), "data", "DualLabel")
    
    print("Testing dataset loading...")
    
    # Create unified dataset
    full_dataset = SegmentationDataset(data_path, img_size=512, val_size=0.2)
    
    # Get train and val splits
    train_dataset = full_dataset.get_train_dataset(augment=True)
    val_dataset = full_dataset.get_val_dataset(augment=False)
    
    print(f"Train set: {len(train_dataset)} images")
    print(f"Val set: {len(val_dataset)} images")
    
    # Verify no overlap
    train_indices = set(train_dataset.indices)
    val_indices = set(val_dataset.indices)
    overlap = train_indices.intersection(val_indices)
    print(f"Overlap between train/val: {len(overlap)} images (should be 0)")
    
    # Test loading a sample
    if len(train_dataset) > 0:
        image, mask = train_dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Unique classes in mask: {torch.unique(mask)}")
        print(f"Mask dtype: {mask.dtype}")
    else:
        print("No images found in dataset!")
