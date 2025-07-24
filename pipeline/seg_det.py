import os
import time

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box
from supervision.utils.conversion import pillow_to_cv2
from tqdm import tqdm
from ultralytics import YOLO

# ----------------- Configurations -----------------
USE_YOLO_DETECTOR = True # Set to True to use YOLO for detection, else use RF-DETR
SEG_MODEL_PATH = os.path.join(os.getcwd(), "yolo_seg", "best-glad-sound-59.pt")
DET_MODEL_PATH = os.path.join(os.getcwd(), "yolo_det", "best-lively-durian-49.pt") if USE_YOLO_DETECTOR else os.path.join(os.getcwd(), "rf_detr", "checkpoint_best_ema_drawn-wind-91.pth")
SEG_CONF = 0.50 # Confidence threshold for segmentation
DET_CONF = 0.25 # Confidence threshold for detection
YOLO_IMG_SZ = 1280 # Image size for inference with YOLO
DEVICE = "cuda" # Device to use for inference (e.g., "cuda" or "cpu")
AREA_OVERLAP_THRESHOLD = 0.04  # how much of the tooth‐mask must be covered to call it "affected"? e.g. 4% of the mask area
VERTICAL_ROI_EXPANSION = 0.3  # How much to expand the ROI vertically (30% of the height) (alpha)
HORIZONTAL_ROI_EXPANSION = 0.05  # How much to expand the ROI horizontally (5% of the width) (beta)
DATA_DIR = os.path.join(os.getcwd(), "data", "InferenceData")
FINAL_DATA_DIR = os.path.join(DATA_DIR, "final")

# Create directories for saving masks AND detections
MASK_COMPARISON_DIR = os.path.join(DATA_DIR, "mask_comparison")
DETECTION_COMPARISON_DIR = os.path.join(DATA_DIR, "detection_comparison")

FULL_MASKS_DIR = os.path.join(MASK_COMPARISON_DIR, "full_image_masks")
ROI_MASKS_DIR = os.path.join(MASK_COMPARISON_DIR, "roi_based_masks")
OVERLAY_MASKS_DIR = os.path.join(MASK_COMPARISON_DIR, "overlay_comparison")

FULL_DETECTIONS_DIR = os.path.join(DETECTION_COMPARISON_DIR, "full_image_detections")
ROI_DETECTIONS_DIR = os.path.join(DETECTION_COMPARISON_DIR, "roi_based_detections")
OVERLAY_DETECTIONS_DIR = os.path.join(DETECTION_COMPARISON_DIR, "overlay_comparison")

def create_expanded_roi_from_masks(masks, img_width, img_height, expansion_factors=(0.3, 0.1)):
    """
    Create expanded ROIs from segmentation masks to capture PAI lesions
    that might be located above or below teeth.
    
    Args:
        masks: List of mask dictionaries with 'mask' polygon coordinates
        img_width, img_height: Image dimensions
        expansion_factors: Tuple of expansion factors for vertical and horizontal expansion (how much to expand the ROI)
    
    Returns:
        Combined expanded mask as binary image
    """
    # We simply look at the box coordinates for every mask to denote the top-left, top-right, bottom-right, bottom-left coordinates
    x_min, y_min, x_max, y_max = -1, -1, -1, -1
    
    # Iterate through masks to find the bounding box coordinates
    for m in masks:
        if x_min == -1 or m['box'][0] < x_min:
            x_min = m['box'][0]
        if y_min == -1 or m['box'][1] < y_min:
            y_min = m['box'][1]
        if x_max == -1 or m['box'][2] > x_max:
            x_max = m['box'][2]
        if y_max == -1 or m['box'][3] > y_max:
            y_max = m['box'][3]

    # Calculate the overall bounding box from all masks
    x = x_min
    y = y_min
    w = x_max - x_min
    h = y_max - y_min

    # Expand the bounding box, especially vertically for PAI lesions
    vertical_expansion = int(h * expansion_factors[0])      # Use h (height) for vertical expansion
    horizontal_expansion = int(w * expansion_factors[1])    # Use w (width) for horizontal expansion

    x_expanded = max(0, x - horizontal_expansion)
    y_expanded = max(0, y - vertical_expansion)
    w_expanded = min(img_width - x_expanded, w + 2 * horizontal_expansion)
    h_expanded = min(img_height - y_expanded, h + 2 * vertical_expansion)

    # Create a mask for the expanded ROI
    expanded_roi = np.zeros((img_height, img_width), dtype=np.uint8)

    # Create expanded rectangular ROI
    cv2.rectangle(expanded_roi, 
                (x_expanded, y_expanded), 
                (x_expanded + w_expanded, y_expanded + h_expanded), 
                255, -1)
    
    return expanded_roi

def run_segmentation_on_roi(image_path, roi_mask, segmentation_model, conf_threshold, device):
    """
    Run segmentation only on the ROI defined by the mask
    """
    if isinstance(image_path, str):
        original_image = cv2.imread(image_path)
    else:
        original_image = image_path
    
    if roi_mask is not None:
        # Find ROI bounding box for cropping
        coords = np.column_stack(np.where(roi_mask > 0))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Crop image to ROI
            cropped_image = original_image[y_min:y_max, x_min:x_max]
            
            # Run segmentation on cropped image
            roi_start = time.time()
            results = segmentation_model.predict(cropped_image, conf=conf_threshold, device=device, retina_masks=True, verbose=False)
            roi_end = time.time()
            
            # Adjust coordinates back to original image space
            if results[0].masks:
                adjusted_masks = []
                for mask_xy, box in zip(results[0].masks.xy, results[0].boxes):
                    # Adjust mask coordinates
                    adjusted_coords = [(x + x_min, y + y_min) for x, y in mask_xy.tolist()]
                    
                    # Adjust box coordinates
                    adjusted_box = box.xyxy.cpu().squeeze().numpy().astype(int)
                    adjusted_box[0] += x_min  # x1
                    adjusted_box[1] += y_min  # y1
                    adjusted_box[2] += x_min  # x2
                    adjusted_box[3] += y_min  # y2
                    
                    adjusted_masks.append({
                        'mask': adjusted_coords,
                        'box': adjusted_box.tolist(),
                        'class_id': int(box.cls),
                    })
                return adjusted_masks, roi_end - roi_start
            
    return []

def run_detection_on_roi(image_path, roi_mask, detector, conf_threshold, device):
    """
    Run detection only on the ROI defined by the mask
    
    Args:
        image_path: Path to the image
        roi_mask: Binary mask defining the ROI
        detector: YOLO or RF-DETR detector
        conf_threshold: Confidence threshold
        device: Device to run inference on
    
    Returns:
        Detection results
    """
    # Load original image
    if isinstance(image_path, str):
        original_image = cv2.imread(image_path)
    else:
        original_image = image_path
    
    # Apply ROI mask to the image
    if roi_mask is not None:
        masked_image = cv2.bitwise_and(original_image, original_image, mask=roi_mask)
    else:
        masked_image = original_image.copy()
    
    # Run detection on the masked image
    if hasattr(detector, 'predict') and 'yolo' in str(type(detector)).lower():
        # YOLO detector
        results = detector.predict(masked_image, conf=conf_threshold, device=device, verbose=False)
        return results
    else:
        # RF-DETR detector
        # Convert to PIL Image for RF-DETR
        masked_pil = Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
        results = detector.predict(masked_pil, threshold=conf_threshold)
        return results

def detect_dental_arch_roi(image, roi_factor=0.75):
    h, w = image.shape[:2]
    
    # For panoramic X-rays, teeth are typically in the middle 60-70% vertically
    # and across most of the width
    roi_height = int(h * roi_factor)
    roi_y_start = int(h * (1 - roi_factor) / 2)
    
    # Create coarse ROI mask
    coarse_roi = np.zeros((h, w), dtype=np.uint8)
    coarse_roi[roi_y_start:roi_y_start + roi_height, :] = 255
    
    return coarse_roi, (0, roi_y_start, w, roi_height)

def save_segmentation_masks(img_file, original_image, full_results, roi_results, save_dirs):
    """
    Save segmentation masks for comparison between Full-image and ROI-based approaches
    """
    img_height, img_width = original_image.shape[:2]
    
    # Create mask images
    full_mask_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    roi_mask_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    overlay_img = original_image.copy()
    
    # Process full-image results
    full_teeth_count = 0
    if full_results[0].masks:
        for mask_xy, box in zip(full_results[0].masks.xy, full_results[0].boxes):
            coords = mask_xy.tolist()
            class_id = int(box.cls)
            
            # Get color for this tooth class
            if class_id in tooth_colors:
                color = tooth_colors[class_id]
                full_teeth_count += 1
                
                # Draw on full mask image
                coords_array = np.array([[[int(x), int(y)] for x, y in coords]], dtype=np.int32)
                cv2.fillPoly(full_mask_img, coords_array, color)
                cv2.polylines(full_mask_img, coords_array, True, (255, 255, 255), 2)
                
                # Add tooth number label
                center_x = int(np.mean([x for x, y in coords]))
                center_y = int(np.mean([y for x, y in coords]))
                cv2.putText(full_mask_img, tooth_numbers_map[class_id], 
                           (center_x-10, center_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Process ROI-based results
    roi_teeth_count = 0
    if roi_results:
        for mask_data in roi_results:
            coords = mask_data['mask']
            class_id = mask_data['class_id']
            
            # Get color for this tooth class
            if class_id in tooth_colors:
                color = tooth_colors[class_id]
                roi_teeth_count += 1
                
                # Draw on ROI mask image
                coords_array = np.array([[[int(x), int(y)] for x, y in coords]], dtype=np.int32)
                cv2.fillPoly(roi_mask_img, coords_array, color)
                cv2.polylines(roi_mask_img, coords_array, True, (255, 255, 255), 2)
                
                # Add tooth number label
                center_x = int(np.mean([x for x, y in coords]))
                center_y = int(np.mean([y for x, y in coords]))
                cv2.putText(roi_mask_img, tooth_numbers_map[class_id], 
                           (center_x-10, center_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Create overlay comparison
    # Full image masks in red channel, ROI masks in green channel
    full_gray = cv2.cvtColor(full_mask_img, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.cvtColor(roi_mask_img, cv2.COLOR_BGR2GRAY)
    
    overlay_comparison = overlay_img.copy()
    # Add full masks in red
    overlay_comparison[:, :, 2] = np.maximum(overlay_comparison[:, :, 2], full_gray)
    # Add ROI masks in green  
    overlay_comparison[:, :, 1] = np.maximum(overlay_comparison[:, :, 1], roi_gray)
    
    # Add legend text
    cv2.putText(overlay_comparison, f"Red: Full-img ({full_teeth_count} teeth)", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(overlay_comparison, f"Green: ROI-based ({roi_teeth_count} teeth)", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(overlay_comparison, f"Yellow: Both methods", 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Add teeth count difference
    teeth_diff = full_teeth_count - roi_teeth_count
    diff_color = (0, 0, 255) if teeth_diff > 0 else (0, 255, 0) if teeth_diff < 0 else (255, 255, 255)
    cv2.putText(overlay_comparison, f"Difference: {teeth_diff:+d} teeth", 
               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, diff_color, 2)
    
    # Save all images
    base_name = img_file.replace('.jpg', '').replace('.png', '')
    
    cv2.imwrite(os.path.join(save_dirs['full'], f"{base_name}_full_masks.png"), full_mask_img)
    cv2.imwrite(os.path.join(save_dirs['roi'], f"{base_name}_roi_masks.png"), roi_mask_img)
    cv2.imwrite(os.path.join(save_dirs['overlay'], f"{base_name}_comparison.png"), overlay_comparison)
    
    return full_teeth_count, roi_teeth_count, teeth_diff

def save_detection_masks(img_file, original_image, full_det_results, roi_det_results, save_dirs):
    """
    Save detection masks for comparison between Full-image and ROI-based PAI lesion detection
    """
    img_height, img_width = original_image.shape[:2]
    
    # Create detection images
    full_det_img = original_image.copy()
    roi_det_img = original_image.copy()
    overlay_comparison = original_image.copy()
    
    # Colors for different PAI levels
    pai_colors = {
        0: (255, 0, 0),    # PAI 3 - Blue
        1: (0, 255, 0),    # PAI 4 - Green  
        2: (0, 0, 255),    # PAI 5 - Red
    }
    
    # Process full-image detection results
    full_lesion_count = 0
    full_detections = []
    
    if USE_YOLO_DETECTOR and full_det_results[0].boxes is not None:
        for box in full_det_results[0].boxes:
            cid = int(box.cls.cpu().numpy()[0])
            name = full_det_results[0].names[cid]
            xy = box.xyxy.cpu().squeeze().numpy().astype(int).tolist()
            conf = float(box.conf)
            full_detections.append({
                'class_id': cid, 
                'class_name': name, 
                'box': xy, 
                'confidence': conf
            })
            full_lesion_count += 1
            
            # Draw on full detection image
            color = pai_colors.get(cid, (255, 255, 255))
            cv2.rectangle(full_det_img, (xy[0], xy[1]), (xy[2], xy[3]), color, 3)
            cv2.putText(full_det_img, f"{name} ({conf:.2f})", 
                       (xy[0], xy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    elif not USE_YOLO_DETECTOR and full_det_results.xyxy is not None:
        for detection_idx in range(len(full_det_results)):
            cid = int(full_det_results.class_id[detection_idx])
            name = PAI_mapping[cid]
            xyxy = full_det_results.xyxy[detection_idx].astype(int)
            conf = float(full_det_results.confidence[detection_idx])
            full_detections.append({
                'class_id': cid, 
                'class_name': name, 
                'box': xyxy.tolist(),
                'confidence': conf
            })
            full_lesion_count += 1
            
            # Draw on full detection image
            color = pai_colors.get(cid, (255, 255, 255))
            cv2.rectangle(full_det_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 3)
            cv2.putText(full_det_img, f"{name} ({conf:.2f})", 
                       (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Process ROI-based detection results
    roi_lesion_count = len(roi_det_results) if roi_det_results else 0
    
    if roi_det_results:
        for det in roi_det_results:
            cid = det['class_id']
            name = det['class_name']
            xy = det['box']
            conf = det['confidence']
            
            # Draw on ROI detection image
            color = pai_colors.get(cid, (255, 255, 255))
            cv2.rectangle(roi_det_img, (xy[0], xy[1]), (xy[2], xy[3]), color, 3)
            cv2.putText(roi_det_img, f"{name} ({conf:.2f})", 
                       (xy[0], xy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Create overlay comparison
    # Full detections with thicker red border, ROI detections with thicker green border
    for det in full_detections:
        xy = det['box']
        cv2.rectangle(overlay_comparison, (xy[0], xy[1]), (xy[2], xy[3]), (0, 0, 255), 4)  # Red for full
    
    for det in roi_det_results if roi_det_results else []:
        xy = det['box']
        cv2.rectangle(overlay_comparison, (xy[0], xy[1]), (xy[2], xy[3]), (0, 255, 0), 2)  # Green for ROI
    
    # Add legend text
    cv2.putText(overlay_comparison, f"Red: Full-img ({full_lesion_count} lesions)", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(overlay_comparison, f"Green: ROI-based ({roi_lesion_count} lesions)", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(overlay_comparison, f"Yellow: Both methods detect same area", 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Add lesion count difference
    lesion_diff = full_lesion_count - roi_lesion_count
    diff_color = (0, 0, 255) if lesion_diff > 0 else (0, 255, 0) if lesion_diff < 0 else (255, 255, 255)
    cv2.putText(overlay_comparison, f"Difference: {lesion_diff:+d} lesions", 
               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, diff_color, 2)
    
    # Add PAI level breakdown
    full_pai_counts = {0: 0, 1: 0, 2: 0}
    roi_pai_counts = {0: 0, 1: 0, 2: 0}
    
    for det in full_detections:
        full_pai_counts[det['class_id']] += 1
    
    for det in roi_det_results if roi_det_results else []:
        roi_pai_counts[det['class_id']] += 1
    
    y_offset = 150
    for pai_level in [0, 1, 2]:
        pai_name = PAI_mapping[pai_level]
        full_count = full_pai_counts[pai_level]
        roi_count = roi_pai_counts[pai_level]
        color = pai_colors[pai_level]
        
        cv2.putText(overlay_comparison, f"{pai_name}: Full={full_count}, ROI={roi_count}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 25
    
    # Save all images
    base_name = img_file.replace('.jpg', '').replace('.png', '')
    
    cv2.imwrite(os.path.join(save_dirs['full'], f"{base_name}_full_detections.png"), full_det_img)
    cv2.imwrite(os.path.join(save_dirs['roi'], f"{base_name}_roi_detections.png"), roi_det_img)
    cv2.imwrite(os.path.join(save_dirs['overlay'], f"{base_name}_detection_comparison.png"), overlay_comparison)
    
    return full_lesion_count, roi_lesion_count, lesion_diff, full_pai_counts, roi_pai_counts

# --- Color and Tooth Mapping ---
def generate_tooth_colors(n_classes=33):
    base_colors = [*plt.cm.tab20.colors, *plt.cm.tab20b.colors, *plt.cm.Set3.colors, *plt.cm.Pastel1.colors]
    colors_255 = [tuple(int(255 * x) for x in mcolors.to_rgb(c)) for c in base_colors[:n_classes]]
    # Class index to tooth number mapping
    # For DualLabel, taken from: https://www.123dentist.com/wp-content/uploads/2017/06/teeth-numbering-systems.png
    tooth_names = {
        0: '11', 1: '12', 2: '13', 3: '14', 4: '15', 5: '16', 6: '17', 7: '18', 8: '21',
        9: '22', 10: '23', 11: '24', 12: '25', 13: '26', 14: '27', 15: '28', 16: '31', 17: '32',
        18: '33', 19: '34', 20: '35', 21: '36', 22: '37', 23: '38', 24: '41', 25: '42', 26: '43',
        27: '44', 28: '45', 29: '46', 30: '47', 31: '48', 32: '91'
    }
    return {idx: colors_255[idx] for idx in tooth_names.keys()}, tooth_names

# --- IoU Utility ---
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2
    xi1, yi1 = max(x1, x1p), max(y1, y1p)
    xi2, yi2 = min(x2, x2p), min(y2, y2p)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2p - x1p) * (y2p - y1p)
    union = area1 + area2 - inter
    return inter / union if union else 0

# --- PAI Mapping ---
PAI_mapping = {
    0: "PAI 3",
    1: "PAI 4",
    2: "PAI 5",
}

for dir_path in [MASK_COMPARISON_DIR, DETECTION_COMPARISON_DIR, 
                 FULL_MASKS_DIR, ROI_MASKS_DIR, OVERLAY_MASKS_DIR,
                 FULL_DETECTIONS_DIR, ROI_DETECTIONS_DIR, OVERLAY_DETECTIONS_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Load models
yolo_seg = YOLO(SEG_MODEL_PATH)
obj_detector = YOLO(DET_MODEL_PATH) if USE_YOLO_DETECTOR else RFDETRBase(pretrain_weights=DET_MODEL_PATH, num_classes=2)

# --------------------------------------------------

# Ensure the final data directory exists
if not os.path.exists(FINAL_DATA_DIR):
    os.makedirs(FINAL_DATA_DIR)
    
mask_predictions = {}
det_predictions = {}

# Generate tooth colors and mapping
tooth_colors, tooth_numbers_map = generate_tooth_colors(33)

# --- Inference Loop ---
full_seg_times = []
roi_seg_times = []
full_det_times = []
roi_det_times = []

files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.png'))] # Filter only .jpg and .png files
files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort files by number

print(f"Starting inference on {len(files)} images...")

for img_file in tqdm(files, desc="Processing images"):

    path = os.path.join(DATA_DIR, img_file)
    img = cv2.imread(path)
    img_height, img_width = img.shape[:2]
    
    # ===== FULL IMAGE APPROACH =====
    # Measure full-image segmentation time
    full_start = time.time()
    full_seg_res = yolo_seg.predict(path, imgsz=YOLO_IMG_SZ, conf=SEG_CONF, device=DEVICE, retina_masks=True, verbose=False)
    full_seg_time = time.time() - full_start
    full_seg_times.append(full_seg_time)
    
    # Measure full-image detection time
    full_det_start = time.time()
    if USE_YOLO_DETECTOR:
        full_det_res = obj_detector.predict(path, conf=DET_CONF, device=DEVICE, verbose=False)
    else:
        full_img_pil = Image.open(path)
        full_det_res = obj_detector.predict(full_img_pil, threshold=DET_CONF)
    full_det_time = time.time() - full_det_start
    full_det_times.append(full_det_time)
    
    # ===== ROI-BASED APPROACH =====    
    # Create adaptive ROI
    dental_roi_mask, roi_bbox = detect_dental_arch_roi(img)
    
    # Run segmentation on ROI
    roi_mask_predictions, roi_seg_time = run_segmentation_on_roi(img, dental_roi_mask, yolo_seg, SEG_CONF, DEVICE)
    roi_seg_times.append(roi_seg_time)
       
    # Calculate area reduction
    roi_area = np.count_nonzero(dental_roi_mask)
    total_area = img_height * img_width
    area_reduction = (1 - roi_area / total_area) * 100
    
    print(f"Image: {img_file}")
    print(f"Full image segmentation time: {full_seg_time:.4f}s")
    print(f"ROI-based segmentation time: {roi_seg_time:.4f}s")
    print(f"Segmentation speedup: {full_seg_time/roi_seg_time:.2f}x")
    print(f"Full image detection time: {full_det_time:.4f}s")

    # Save mask comparison
    save_dirs = {
        'full': FULL_MASKS_DIR,
        'roi': ROI_MASKS_DIR, 
        'overlay': OVERLAY_MASKS_DIR
    }
    
    full_count, roi_count, diff = save_segmentation_masks(
        img_file, img, full_seg_res, roi_mask_predictions, save_dirs
    )
    
    print(f"Teeth detected - Full: {full_count}, ROI: {roi_count}, Difference: {diff:+d}")
 
    if roi_mask_predictions:
        # Create expanded ROI mask from the segmentation masks
        expanded_roi_mask = create_expanded_roi_from_masks(roi_mask_predictions, img_width, img_height, expansion_factors=(VERTICAL_ROI_EXPANSION, HORIZONTAL_ROI_EXPANSION))
        
        # Store for later use in visualization
        mask_predictions[img_file] = roi_mask_predictions
        
        # ===== Detection on ROI
        roi_det_start = time.time()
        det_res = run_detection_on_roi(img, expanded_roi_mask, obj_detector, DET_CONF, DEVICE)
        roi_det_time = time.time() - roi_det_start
        roi_det_times.append(roi_det_time)
        
        print(f"ROI-based detection time: {roi_det_time:.4f}s")
        print(f"Detection speedup: {full_det_time/roi_det_time:.2f}x")
        
        # Handle results based on detector type
        if USE_YOLO_DETECTOR:
            # YOLO returns a list with Results objects
            det_boxes = det_res[0].boxes
            if det_boxes:
                det_predictions[img_file] = []
                for box in det_boxes:
                    cid = int(box.cls.cpu().numpy()[0])
                    name = det_res[0].names[cid]
                    xy = box.xyxy.cpu().squeeze().numpy().astype(int).tolist()
                    det_predictions[img_file].append({
                        'class_id': cid, 
                        'class_name': name, 
                        'box': xy, 
                        'confidence': float(box.conf)
                    })
        else:
            # RF-DETR returns a Detection object
            if det_res.xyxy is not None:
                det_predictions[img_file] = []
                for detection_idx in range(len(det_res)):
                    cid = int(det_res.class_id[detection_idx])
                    name = PAI_mapping[cid]
                    xyxy = det_res.xyxy[detection_idx].astype(int)
                    det_predictions[img_file].append({
                        'class_id': cid, 
                        'class_name': name, 
                        'box': xyxy,
                        'confidence': float(det_res.confidence[detection_idx])
                    })
    else:
        # If no ROI masks found, add 0 to maintain array consistency
        roi_det_times.append(0)
        print("No ROI masks found - skipping ROI detection")
 
     # Save detection comparison
    save_dirs_detections = {
        'full': FULL_DETECTIONS_DIR,
        'roi': ROI_DETECTIONS_DIR, 
        'overlay': OVERLAY_DETECTIONS_DIR
    }
    
    # Get ROI detection results for comparison
    roi_det_results = det_predictions.get(img_file, [])
    
    full_lesion_count, roi_lesion_count, lesion_diff, full_pai_counts, roi_pai_counts = save_detection_masks(
        img_file, img, full_det_res, roi_det_results, save_dirs_detections
    )
    
    print(f"PAI lesions detected - Full: {full_lesion_count}, ROI: {roi_lesion_count}, Difference: {lesion_diff:+d}")
    print(f"Full PAI breakdown - PAI3: {full_pai_counts[0]}, PAI4: {full_pai_counts[1]}, PAI5: {full_pai_counts[2]}")
    print(f"ROI PAI breakdown - PAI3: {roi_pai_counts[0]}, PAI4: {roi_pai_counts[1]}, PAI5: {roi_pai_counts[2]}")
    
    # Calculate combined times
    full_combined_time = full_seg_time + full_det_time
    roi_combined_time = roi_seg_time + roi_det_times[-1] if roi_det_times[-1] > 0 else roi_seg_time
    
    print(f"Full pipeline time: {full_combined_time:.4f}s")
    print(f"ROI pipeline time: {roi_combined_time:.4f}s")
    print(f"Overall speedup: {full_combined_time/roi_combined_time:.2f}x")
    print(f"Area reduction: {area_reduction:.1f}%")
    print("-" * 50)
    
    print("Found {} PAI lesions in {}.".format(len(det_predictions.get(img_file, [])), img_file))

# --- Match and Visualize with Supervision ---
# Sort mask_predictions by file name
mask_predictions = dict(sorted(mask_predictions.items(), key=lambda x: int(x[0].split('_')[1].split('.')[0])))

for img_file, masks in mask_predictions.items():
    detections = det_predictions.get(img_file, [])
    
    if not detections: # No detections for this image
        continue
    
    # Process overlaps between masks and detections
    matches = []
    for m in masks:
        mask_poly = Polygon(m['mask'])
        mask_area = mask_poly.area
        
        if mask_area == 0:
            continue

        for d in detections:
            x1, y1, x2, y2 = d['box']
            box_poly = shapely_box(x1, y1, x2, y2)

            # Calculate overlap area
            inter_area = mask_poly.intersection(box_poly).area
            frac = inter_area / mask_area  # fraction of tooth area that's affected

            if frac >= AREA_OVERLAP_THRESHOLD:
                matches.append({
                    'mask': m['mask'],
                    'mask_class_id': m['class_id'],
                    'tooth': tooth_numbers_map[m['class_id']],
                    'box': d['box'],
                    'disease': d['class_name'],
                    'disease_class_id': d['class_id'],
                    'confidence': d['confidence'],
                    'overlap': frac,
                    'mask_color': tooth_colors[m['class_id']],
                })
    
    # Only proceed if matches are found
    if not matches:
        print(f"No matches found for {img_file}.")
        continue
    
    # Load original image
    original_image = Image.open(os.path.join(DATA_DIR, img_file))
    img_width, img_height = original_image.size
    
    # Convert to OpenCV format for Supervision
    image_cv = pillow_to_cv2(original_image)
    
    # Create annotators
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(img_width, img_height))
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(img_width, img_height))
    
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=thickness,
        smart_position=True
    )
    mask_annotator = sv.MaskAnnotator(opacity=0.3)
    
    # Prepare detections for Supervision
    boxes = []
    class_ids = []
    confidences = []
    labels = []
    
    # Convert masks to a format that Supervision can use
    mask_tensors = []
    
    for item in matches:
        # Box
        boxes.append(item['box'])
        class_ids.append(item['disease_class_id'])
        confidences.append(item['confidence'])
        
        # Create label with tooth number, PAI value and overlap
        label = f"{item['tooth']} - {item['disease']} - Overlap: {item['overlap']:.2f}"
        labels.append(label)
        
        # Create mask
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        coords = np.array([[[int(x), int(y)] for x, y in item['mask']]], dtype=np.int32)
        cv2.fillPoly(mask, coords, 1)
        mask_tensors.append(mask)
    
    # Convert to numpy arrays for Supervision
    boxes = np.array(boxes)
    class_ids = np.array(class_ids)
    confidences = np.array(confidences)
    
    # Create Detections object
    detections = sv.Detections(
        xyxy=boxes,
        confidence=confidences,
        class_id=class_ids
    )
    
    # Create mask detections
    if mask_tensors:
        masks = np.stack(mask_tensors, axis=0)
        detections.mask = masks
    
    # Annotate
    annotated_image = image_cv.copy()
    
    # Draw masks with class-specific colors
    for i, (item, mask) in enumerate(zip(matches, mask_tensors)):
        rgb_color = item['mask_color']
        # Create a Supervision Color object instead of a tuple
        color = sv.Color(rgb_color[0], rgb_color[1], rgb_color[0])
        
        # Convert polygon points to the correct format for OpenCV
        polygon_points = np.array(item['mask'], dtype=np.int32)
        
        # Check if polygon has valid points and reshape properly
        if len(polygon_points) >= 3:  # Need at least 3 points for a valid polygon
            # Draw polygon outline
            cv2.polylines(
                annotated_image, 
                [polygon_points], 
                isClosed=True, 
                color=(int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])), 
                thickness=2
            )
            
            # For the filled polygon, convert to BGR for OpenCV
            bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
            mask_overlay = annotated_image.copy()
            cv2.fillPoly(mask_overlay, [polygon_points], bgr_color)
            annotated_image = cv2.addWeighted(mask_overlay, 0.3, annotated_image, 0.7, 0)
    
    # Add bounding boxes
    annotated_image = box_annotator.annotate(annotated_image, detections)
    
    # Add labels
    annotated_image = label_annotator.annotate(annotated_image, detections, labels)
    
    # Save the result
    save_path = os.path.join(FINAL_DATA_DIR, img_file.replace('.', '_final.'))
    cv2.imwrite(save_path, annotated_image)
    print(f"Saved: {save_path}")

# Calculate averages
avg_full_seg = sum(full_seg_times) / len(full_seg_times) if full_seg_times else 0
avg_roi_seg = sum(roi_seg_times) / len(roi_seg_times) if roi_seg_times else 0
avg_full_det = sum(full_det_times) / len(full_det_times) if full_det_times else 0
avg_roi_det = sum([t for t in roi_det_times if t > 0]) / len([t for t in roi_det_times if t > 0]) if any(t > 0 for t in roi_det_times) else 0

# Calculate speedups
seg_speedup = avg_full_seg / avg_roi_seg if avg_roi_seg > 0 else 0
det_speedup = avg_full_det / avg_roi_det if avg_roi_det > 0 else 0

# Combined pipeline times
avg_full_combined = avg_full_seg + avg_full_det
avg_roi_combined = avg_roi_seg + avg_roi_det
overall_speedup = avg_full_combined / avg_roi_combined if avg_roi_combined > 0 else 0

print("\n" + "="*60)
print("PERFORMANCE ANALYSIS")
print("="*60)
print(f"SEGMENTATION:")
print(f"  Full Image Average: {avg_full_seg:.4f}s")
print(f"  ROI-based Average:  {avg_roi_seg:.4f}s")
print(f"  Speedup:           {seg_speedup:.2f}x")
print()
print(f"DETECTION:")
print(f"  Full Image Average: {avg_full_det:.4f}s")
print(f"  ROI-based Average:  {avg_roi_det:.4f}s")
print(f"  Speedup:           {det_speedup:.2f}x")
print()
print(f"COMBINED PIPELINE:")
print(f"  Full Image Average: {avg_full_combined:.4f}s")
print(f"  ROI-based Average:  {avg_roi_combined:.4f}s")
print(f"  Overall Speedup:   {overall_speedup:.2f}x")
print()
print(f"PROCESSED {len(files)} IMAGES")
print(f"Average area reduction: {sum([roi_area/total_area for roi_area, total_area in zip([np.count_nonzero(detect_dental_arch_roi(cv2.imread(os.path.join(DATA_DIR, f)))[0]) for f in files], [cv2.imread(os.path.join(DATA_DIR, f)).shape[0] * cv2.imread(os.path.join(DATA_DIR, f)).shape[1] for f in files])]) / len(files) * 100:.1f}%")

# --- Summary of Mask Comparisons ---
print("\n" + "="*60)
print("MASK COMPARISON SUMMARY")
print("="*60)

# Analyze all saved comparisons
full_counts = []
roi_counts = []
differences = []

for img_file in files:
    # Re-process to get counts (you could also store these during the main loop)
    path = os.path.join(DATA_DIR, img_file)
    img = cv2.imread(path)
    
    # Full image segmentation count
    full_seg_res = yolo_seg.predict(path, imgsz=YOLO_IMG_SZ, conf=SEG_CONF, device=DEVICE, retina_masks=True, verbose=False)
    full_count = len(full_seg_res[0].masks) if full_seg_res[0].masks else 0
    
    # ROI segmentation count
    roi_count = len(mask_predictions.get(img_file, []))
    
    full_counts.append(full_count)
    roi_counts.append(roi_count)
    differences.append(full_count - roi_count)

avg_full_teeth = sum(full_counts) / len(full_counts) if full_counts else 0
avg_roi_teeth = sum(roi_counts) / len(roi_counts) if roi_counts else 0
avg_difference = sum(differences) / len(differences) if differences else 0

print(f"Average teeth detected:")
print(f"  Full Image: {avg_full_teeth:.1f}")
print(f"  ROI-based:  {avg_roi_teeth:.1f}")
print(f"  Difference: {avg_difference:+.1f}")
print()

# Show distribution of differences
from collections import Counter

diff_counts = Counter(differences)
print("Teeth detection differences distribution:")
for diff in sorted(diff_counts.keys()):
    count = diff_counts[diff]
    percentage = (count / len(differences)) * 100
    print(f"  {diff:+2d} teeth: {count:2d} images ({percentage:.1f}%)")

print(f"\nMask comparison images saved to: {MASK_COMPARISON_DIR}")

print("\n" + "="*60)
print("DETECTION COMPARISON SUMMARY")
print("="*60)

# Analyze all detection comparisons
full_lesion_counts = []
roi_lesion_counts = []
lesion_differences = []
full_pai_totals = {0: 0, 1: 0, 2: 0}
roi_pai_totals = {0: 0, 1: 0, 2: 0}

for img_file in files:
    # Re-process to get detection counts
    path = os.path.join(DATA_DIR, img_file)
    
    # Full image detection count
    if USE_YOLO_DETECTOR:
        full_det_res = obj_detector.predict(path, conf=DET_CONF, device=DEVICE, verbose=False)
        full_lesion_count = len(full_det_res[0].boxes) if full_det_res[0].boxes is not None else 0
        
        # Count PAI levels for full image
        if full_det_res[0].boxes is not None:
            for box in full_det_res[0].boxes:
                cid = int(box.cls.cpu().numpy()[0])
                full_pai_totals[cid] += 1
    else:
        full_img_pil = Image.open(path)
        full_det_res = obj_detector.predict(full_img_pil, threshold=DET_CONF)
        full_lesion_count = len(full_det_res) if full_det_res.xyxy is not None else 0
        
        # Count PAI levels for full image
        if full_det_res.xyxy is not None:
            for detection_idx in range(len(full_det_res)):
                cid = int(full_det_res.class_id[detection_idx])
                full_pai_totals[cid] += 1
    
    # ROI detection count
    roi_lesion_count = len(det_predictions.get(img_file, []))
    
    # Count PAI levels for ROI
    for det in det_predictions.get(img_file, []):
        roi_pai_totals[det['class_id']] += 1
    
    full_lesion_counts.append(full_lesion_count)
    roi_lesion_counts.append(roi_lesion_count)
    lesion_differences.append(full_lesion_count - roi_lesion_count)

avg_full_lesions = sum(full_lesion_counts) / len(full_lesion_counts) if full_lesion_counts else 0
avg_roi_lesions = sum(roi_lesion_counts) / len(roi_lesion_counts) if roi_lesion_counts else 0
avg_lesion_difference = sum(lesion_differences) / len(lesion_differences) if lesion_differences else 0

print(f"Average PAI lesions detected:")
print(f"  Full Image: {avg_full_lesions:.1f}")
print(f"  ROI-based:  {avg_roi_lesions:.1f}")
print(f"  Difference: {avg_lesion_difference:+.1f}")
print()

# PAI level breakdown
print("PAI Level Distribution:")
total_full = sum(full_pai_totals.values())
total_roi = sum(roi_pai_totals.values())

for pai_level in [0, 1, 2]:
    pai_name = PAI_mapping[pai_level]
    full_count = full_pai_totals[pai_level]
    roi_count = roi_pai_totals[pai_level]
    full_pct = (full_count / total_full * 100) if total_full > 0 else 0
    roi_pct = (roi_count / total_roi * 100) if total_roi > 0 else 0
    
    print(f"  {pai_name}:")
    print(f"    Full Image: {full_count} ({full_pct:.1f}%)")
    print(f"    ROI-based:  {roi_count} ({roi_pct:.1f}%)")

# Show distribution of detection differences
print("\nPAI lesion detection differences distribution:")
lesion_diff_counts = Counter(lesion_differences)
for diff in sorted(lesion_diff_counts.keys()):
    count = lesion_diff_counts[diff]
    percentage = (count / len(lesion_differences)) * 100
    print(f"  {diff:+2d} lesions: {count:2d} images ({percentage:.1f}%)")

print(f"\nDetection comparison images saved to: {DETECTION_COMPARISON_DIR}")
