import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon  # For centroid calculation
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = os.path.join(os.getcwd(), "yolo_seg", "best.pt")
CONF_THRESHOLD_PRED = 0.30 # Initial confidence threshold for YOLO prediction
IMAGE_SIZE = 1280 # Image size for YOLO model
DEVICE = "cuda" # or "cpu"

# ========================
# > TO BE IMPLEMENTED
# Test-Time Augmentation (TTA) settings
TTA_ANGLES = [0, 15, -15, 30, -30] # Rotation angles for TTA (15deg steps)
IOU_THRESHOLD = 0.75 # IoU threshold for matching masks 
# ========================

IMAGE_PATH = os.path.join(os.getcwd(), "data", "InferenceData", "inference_1.jpg")
SAVE_OUTPUT = False # Set to True to save the image, False to display

# --- Color and Class Definitions (from original script) ---
def generate_tooth_colors(n_classes=33):
    base_colors = [
        *plt.cm.tab20.colors, *plt.cm.tab20b.colors,
        *plt.cm.Set3.colors, *plt.cm.Pastel1.colors
    ]
    colors_255 = [
        tuple(int(255 * x) for x in mcolors.to_rgb(color))
        for color in base_colors[:n_classes]
    ]
    # Class index to tooth number mapping (ensure this matches your model's training)
    tooth_names = {
        0: '13', 1: '14', 2: '15', 3: '11', 4: '12', 5: '19', 6: '20', 7: '21', 8: '22',
        9: '23', 10: '24', 11: '25', 12: '27', 13: '32', 14: '16', 15: '26', 16: '17', 17: '1',
        18: '2', 19: '3', 20: '4', 21: '5', 22: '6', 23: '7', 24: '8', 25: '9', 26: '10',
        27: '18', 28: '28', 29: '29', 30: '30', 31: '31', 32: '13(polygon)'
    }
    # Map class index to color
    return {idx: colors_255[idx] for idx in tooth_names.keys()}, tooth_names

tooth_colors, tooth_numbers_map = generate_tooth_colors(33)

# --- TTA Prediction Function ---
def predict(model, img_path, angles, device, imgsz, conf):
    """Performs prediction with Test-Time Augmentation (rotation)."""
    original_img = Image.open(img_path).convert("RGB")
    img_w, img_h = original_img.size
    center = (img_w / 2, img_h / 2)

    all_predictions = [] # List to store predictions from all angles

    # Predict on image
    results = model.predict(
        original_img,
        imgsz=imgsz,
        conf=conf,
        device=device,
        retina_masks=True,
        verbose=False # Reduce console spam
    )

    # Process results if masks are found
    if results[0].masks is not None:
        masks = results[0].masks.xy # List of numpy arrays (N, 2)
        boxes = results[0].boxes

        for mask_coords, box in zip(masks, boxes):

            prediction = {
                "mask": mask_coords,
                "confidence": float(box.conf),
                "class_id": int(box.cls),
            }
            all_predictions.append(prediction)
    else:
        print(f"No masks found in the original image.")

    return all_predictions

# --- Updated Visualization Function ---
def visualize_final_segmentation(final_predictions, img_path, save_path=None, font=None):
    """Visualize the final selected segmentation masks."""
    img = Image.open(img_path).convert("RGB")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    label_draw = ImageDraw.Draw(img) # Draw labels directly on the image

    print(f"Found {len(final_predictions)} teeth predictions...")
    for pred in final_predictions:
        cls = pred['class_id']
        color = tooth_colors.get(cls, (255, 0, 255)) # Default to magenta if class not found
        mask_coords = [(int(x), int(y)) for x, y in pred['mask']] # Convert to int tuples

        if len(mask_coords) < 3: # Need at least 3 points for a polygon
            print(f"  Skipping invalid mask (less than 3 points) for class {cls}")
            continue

        # Draw segmentation polygon (semi-transparent)
        overlay_draw.polygon(
            mask_coords,
            fill=color + (128,), # Add alpha for transparency
            outline=color + (255,) # Solid outline
        )

        # Get tooth number label
        tooth_number = tooth_numbers_map.get(cls, f"Cls {cls}")
        conf_label = f"{pred['confidence']:.2f}"

        # Draw label near the centroid or first point of the mask
        try:
            poly = Polygon(mask_coords)
            centroid = poly.centroid
            label_pos = (int(centroid.x), int(centroid.y))
        except: # Fallback to first point if centroid fails
             label_pos = mask_coords[0]

        # Simple label background for readability
        text = f"{tooth_number}\n({conf_label})"
        bbox = label_draw.textbbox(label_pos, text, font=font)
        label_draw.rectangle(bbox, fill=(0,0,0,100)) # Slightly transparent black background
        label_draw.text(label_pos, text, fill=color, font=font)

    # Combine overlay with original image
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # Display or save
    if save_path:
        print(f"Saving output image to: {save_path}")
        img.save(save_path)
    else:
        plt.figure(figsize=(16, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title("Dental Segmentation Results", pad=20)
        plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    
    # Load Model
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Perform TTA Prediction
    preds = predict(model, IMAGE_PATH, TTA_ANGLES, DEVICE, IMAGE_SIZE, CONF_THRESHOLD_PRED)

    # Initialize font for labels
    size = 20
    try:
        font = ImageFont.truetype("arial.ttf", size) # Slightly smaller font
    except:
        font = ImageFont.load_default(size)

    # Visualize Final Results
    visualize_final_segmentation(
        final_predictions=preds,
        img_path=IMAGE_PATH,
        save_path=IMAGE_PATH.split(".jpg")[0] + '-output.jpg' if SAVE_OUTPUT else None,
        font=font
    )

    print("Processing finished.")
