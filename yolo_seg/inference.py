from ultralytics import YOLO
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Generate distinct colors for 33 teeth classes
def generate_tooth_colors(n_classes=33):
    # Start with qualitative colormaps that work well for categorical data
    base_colors = [
        *plt.cm.tab20.colors,          # 20 colors
        *plt.cm.tab20b.colors,         # 20 more colors
        *plt.cm.Set3.colors,           # 12 colors
        *plt.cm.Pastel1.colors         # 9 colors
    ]
    
    # Convert to 0-255 RGB range
    colors_255 = [
        tuple(int(255 * x) for x in mcolors.to_rgb(color))
        for color in base_colors[:n_classes]
    ]
    
    # Create mapping to tooth numbers
    tooth_names = {
        0: '13', 1: '14', 2: '15', 3: '11', 4: '12', 5: '19', 6: '20', 7: '21', 8: '22',
        9: '23', 10: '24', 11: '25', 12: '27', 13: '32', 14: '16', 15: '26', 16: '17', 17: '1',
        18: '2', 19: '3', 20: '4', 21: '5', 22: '6', 23: '7', 24: '8', 25: '9', 26: '10',
        27: '18', 28: '28', 29: '29', 30: '30', 31: '31', 32: '13(polygon)'
    }
    
    return {idx: colors_255[idx] for idx in tooth_names.keys()}

# Get our tooth-specific colors
tooth_colors = generate_tooth_colors(33)

# Improved visualization function
def visualize_tooth_segmentation(results, img_path, save_path=None):
    # Load image
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Create transparent overlay
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Get predictions
    masks = results[0].masks
    boxes = results[0].boxes
    
    if masks is not None:
        for mask, box in zip(masks.xy, boxes):
            cls = int(box.cls)
            color = tooth_colors.get(cls, (255, 255, 255))  # Default to white if class not found
            
            # Draw segmentation polygon (50% opacity)
            overlay_draw.polygon(
                [(x, y) for x, y in mask],
                fill=color + (128,),  # Add alpha channel
                outline=color + (255,)  # Solid outline
            )
            
            # Get tooth number from class mapping
            tooth_numbers = {
                0: '13', 1: '14', 2: '15', 3: '11', 4: '12', 5: '19', 6: '20', 7: '21', 8: '22',
                9: '23', 10: '24', 11: '25', 12: '27', 13: '32', 14: '16', 15: '26', 16: '17', 17: '1',
                18: '2', 19: '3', 20: '4', 21: '5', 22: '6', 23: '7', 24: '8', 25: '9', 26: '10',
                27: '18', 28: '28', 29: '29', 30: '30', 31: '31', 32: '13(polygon)'
            }
            tooth_number = tooth_numbers.get(cls, str(cls))
            
            # Draw label at first point
            x, y = mask[0]
            draw.text((x, y), tooth_number, fill=color, font=font)
    
    # Combine overlay with original image
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    
    # Display or save
    if save_path:
        img.save(save_path)
    else:
        plt.figure(figsize=(16, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title("Dental Segmentation Results", pad=20)
        plt.show()

# Initialize model and font
try:
    font = ImageFont.truetype("arial.ttf", 40)
except:
    font = ImageFont.load_default(40)

model = YOLO(os.path.join(os.getcwd(), "yolo_seg", "best.pt"))
img_path = os.path.join(os.getcwd(), "data", "InferenceData", "inference_1.jpg")

# Run prediction with segmentation
results = model.predict(
    img_path,
    imgsz=1280,
    conf=0.25,
    device="cuda",
    retina_masks=True
)

# Visualize results
visualize_tooth_segmentation(results, img_path)
