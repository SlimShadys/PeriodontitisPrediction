import json
import os
import random
import time
from os.path import join as opj

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm

import cv2
import numpy as np

def get_overlapped_img(filename, img_folder, label_folder, classes, color_map):
    """
    Create an overlapped image with segmentation masks from YOLO format labels
    
    Args:
        filename: Image filename without extension
        img_folder: Folder containing original images
        label_folder: Folder containing YOLO format labels
        classes: Dictionary of class information
        color_map: Dictionary mapping class IDs to colors
        
    Returns:
        Overlapped image with segmentation masks
    """
    # Load original image
    img_path = os.path.join(img_folder, f"{filename}.jpg")
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    # Convert to RGB (OpenCV loads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]
    
    # Create a blank mask image (white background)
    mask = Image.new('RGB', (img_width, img_height), (255, 255, 255))
    draw = ImageDraw.Draw(mask)
    
    # Load corresponding label
    label_path = os.path.join(label_folder, f"{filename}.txt")
    if not os.path.exists(label_path):
        return img  # Return original image if no labels
    
    with open(label_path, 'r') as f:
        labels = f.read().splitlines()
    
    # Draw each segmentation polygon on the mask
    for label in labels:
        parts = label.strip().split()
        if len(parts) < 3:  # Need at least class + one point
            continue
        
        class_id = int(parts[0])
        points = list(map(float, parts[1:]))
        
        # Convert normalized to absolute coordinates
        absolute_points = []
        for j in range(0, len(points), 2):
            x = points[j] * img_width
            y = points[j+1] * img_height
            absolute_points.append((x, y))
        
        # Get color for this class (default to magenta if not found)
        color = color_map.get(class_id, (186, 85, 211))  # Magenta
        
        # Draw polygon
        if len(absolute_points) >= 3:
            draw.polygon(absolute_points, fill=color)
    
    # Convert mask to numpy array
    mask_np = np.array(mask)
    
    # Overlay the images with transparency
    overlapped = cv2.addWeighted(img, 0.8, mask_np, 0.3, 0)
    
    return overlapped

# Modified visualization function using the new overlapped image
def visualize_segmentation(directory, examples_to_show=4):
    training_data = opj(directory, 'YOLO_dataset', 'train')
    train_files = [file for file in os.listdir(training_data) if file.endswith('.jpg')]
    
    # Load classes and colors (from your existing code)
    meta = json.load(open(os.path.join(directory, "Teeth Segmentation JSON", "meta.json")))
    obj_class_to_machine_color = json.load(open(os.path.join(directory, "Teeth Segmentation JSON", "obj_class_to_machine_color.json")))
    
    classes = {}
    for (index, entry) in enumerate(meta["classes"]):
        title = entry["title"]
        color = str(entry["color"]).lstrip('#')
        # convert hex to RGB
        color = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        
        classes[title] = (index, color)
    
    color_map = {class_id: tuple(color) for class_id, (_, color) in enumerate(classes.values())}
    
    # Shuffle and select files
    random.seed(time.time())
    random.shuffle(train_files)
    train_files = train_files[:examples_to_show]
    
    # Create figure
    fig, axs = plt.subplots(2, int(examples_to_show/2), figsize=(20, 10))
    axs = axs.flatten()
    
    for i, img_file in enumerate(tqdm(train_files, desc='Processing images')):
        if i >= examples_to_show:
            break
        
        # Get filename without extension
        filename = os.path.splitext(img_file)[0]
        
        # Get overlapped image
        overlapped_img = get_overlapped_img(
            filename=filename,
            img_folder=training_data,
            label_folder=training_data,
            classes=classes,
            color_map=color_map
        )
        
        # Display
        ax = axs[i]
        ax.imshow(overlapped_img)
        ax.set_title(f"Image: {img_file}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Call the visualization function
visualize_segmentation(directory=opj('data', 'TeethSeg'), examples_to_show=4)
