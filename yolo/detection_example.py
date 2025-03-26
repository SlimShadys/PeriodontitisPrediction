import json
import os
import random
import time
from os.path import join as opj

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

directory = opj('data', 'DENTEX', 'DENTEX')

training_data = opj(directory, 'training_data')
testing_data = opj(directory, 'testing_data')
validation_data = opj(directory, 'validation_data')

training_datasets = ['quadrant', 'quadrant_enumeration', 'quadrant-enumeration-disease']

# Simply open an example of the training data (quadrant, quadrant_enumeration)
quadrant_files = os.listdir(opj(training_data, 'quadrant', 'xrays'))
quadrant_json = json.load(open(opj(training_data, 'quadrant', 'train_quadrant.json')))

# Get the corresponding IDs for the quadrants
# Dictionary: {<id>: name_of_the_quadrant}
quadrant_ids = {}
for entry in quadrant_json['categories']:
    if entry['id'] not in quadrant_ids:
        quadrant_ids[entry['id']] = entry['name']

# List of examples
examples_to_show = 8
examples_list = {}

# Shuffle the list of files and get the first K
# get a random seed based on time
random.seed(time.time())
random.shuffle(quadrant_files)
quadrant_files = quadrant_files[:examples_to_show]

# We must cycle through the 'images' tag in the JSON and check if 'file_name' is the same as the random file
for image in tqdm(quadrant_json['images'], desc='Processing images'):
    if image['file_name'] in quadrant_files:
        img_height = image['height']
        img_width = image['width']
        img_id = image['id']
        img_file_name = image['file_name']

        # We know the image ID, so we can find the corresponding annotations
        for annotation in quadrant_json['annotations']:
            if annotation['image_id'] == img_id:
                bbox = annotation['bbox']
                category_id = annotation['category_id'] # This is the quadrant ID which must be converted to the quadrant name through the dictionary
                category_name = quadrant_ids[category_id]
                segmentation = annotation['segmentation']

                print(f'Image ID: {img_id}')
                print(f'Image file name: {img_file_name}')
                print(f'Image height: {img_height}')
                print(f'Image width: {img_width}')
                print(f'Bounding box: {bbox}')
                print(f'Category ID: {category_id}')
                print(f'Quadrant name: {category_name}')
                print("=====================================")

                if img_file_name not in examples_list:
                    examples_list[img_file_name] = []

                examples_list[img_file_name].append((img_file_name, img_height, img_width, bbox, category_name, category_id, segmentation))

# Now we can show the examples by plotting the image and the bounding box
fig, axs = plt.subplots(2, int(examples_to_show/2), figsize=(20, 10))  # Adjusted to 2x3 plot
axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

for i, (img_name, v) in enumerate(examples_list.items()):
    if i >= examples_to_show:  # Ensure we only plot up to 6 images
        break

    ax = axs[i]
    img = Image.open(opj(training_data, 'quadrant', 'xrays', img_name))
    ax.imshow(img)

    # This loops throughout the list of quadrants for the image
    colors = {1: 'r', 2: 'g', 3: 'b', 4: 'y'}

    for _, v in enumerate(v):
        # img_name, img_height, img_width, bbox, category_name, segmentation
        _, _, _, bbox, category_name, category_id, _ = v

        color = colors[int(category_name)]

        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1], category_id, fontsize=12, color=color)

    ax.set_title(f"IMG: {img_name}")
    ax.axis('off')

# plot the window maximized
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

plt.tight_layout()
plt.show()