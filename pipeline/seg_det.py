import os

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

# --- PAI Mapping ---
PAI_mapping = {
    0: "PAI 3",
    1: "PAI 4",
    2: "PAI 5",
}

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

# --- Configurations ---
USE_YOLO_DETECTOR = True # Set to True to use YOLO for detection, else use RF-DETR
SEG_MODEL_PATH = os.path.join(os.getcwd(), "yolo_seg", "best-dulcet-wildflower-40.pt")
DET_MODEL_PATH = os.path.join(os.getcwd(), "yolo_det", "best-lively-durian-49.pt") if USE_YOLO_DETECTOR else os.path.join(os.getcwd(), "rf_detr", "checkpoint_best_ema_drawn-wind-91.pth")
SEG_CONF = 0.50 # Confidence threshold for segmentation
DET_CONF = 0.25 # Confidence threshold for detection
YOLO_IMG_SZ = 1280 # Image size for inference with YOLO
DEVICE = "cuda" # Device to use for inference (e.g., "cuda" or "cpu")
AREA_OVERLAP_THRESHOLD = 0.05  # how much of the tooth‐mask must be covered to call it "affected"? e.g. 5% of the mask area

# Load models
yolo_seg = YOLO(SEG_MODEL_PATH)
obj_detector = YOLO(DET_MODEL_PATH) if USE_YOLO_DETECTOR else RFDETRBase(pretrain_weights=DET_MODEL_PATH, num_classes=2)

DATA_DIR = os.path.join(os.getcwd(), "data", "InferenceData")
FINAL_DATA_DIR = os.path.join(DATA_DIR, "final")
if not os.path.exists(FINAL_DATA_DIR):
    os.makedirs(FINAL_DATA_DIR)
    
mask_predictions = {}
det_predictions = {}

# Generate tooth colors and mapping
tooth_colors, tooth_numbers_map = generate_tooth_colors(33)

# --- Inference Loop ---
files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.png'))] # Filter only .jpg and .png files
files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort files by number

for img_file in tqdm(files, desc="Processing images"):
    path = os.path.join(DATA_DIR, img_file)
    # ===== Segmentation
    seg_res = yolo_seg.predict(path, imgsz=YOLO_IMG_SZ, conf=SEG_CONF, device=DEVICE, retina_masks=True, verbose=False)
    if seg_res[0].masks:
        mask_predictions[img_file] = []
        for mask_xy, box in zip(seg_res[0].masks.xy, seg_res[0].boxes):
            coords = mask_xy.tolist()
            xy = box.xyxy.cpu().squeeze().numpy().astype(int).tolist()
            mask_predictions[img_file].append({
                'mask': coords,
                'box': xy,
                'class_id': int(box.cls),
            })
    
    # ===== Detection
    if USE_YOLO_DETECTOR:
        det_res = obj_detector.predict(path, imgsz=YOLO_IMG_SZ, conf=DET_CONF, device=DEVICE, verbose=False)
        det_boxes = det_res[0].boxes
        if det_boxes:
            det_predictions[img_file] = []
            for box in det_boxes:
                cid = int(box.cls.cpu().numpy()[0])
                name = det_res[0].names[cid]
                xy = box.xyxy.cpu().squeeze().numpy().astype(int).tolist()
                det_predictions[img_file].append({'class_id': cid, 'class_name': name, 'box': xy, 'confidence': float(box.conf)})
    else:
        det_res = obj_detector.predict(path, threshold=DET_CONF)
        if det_res.xyxy is not None:
            idx_to_skip = [] # List to keep track of indices to skip in case of overlapping detections
            if len(det_res.xyxy) > 1:
                # Check if the detections refer to the same area using IoU
                for i in range(len(det_res.xyxy)):
                    for j in range(i + 1, len(det_res.xyxy)):
                        iou = compute_iou(det_res.xyxy[i], det_res.xyxy[j])
                        if iou > 0.95:
                            # Retain the one with the highest confidence
                            if det_res.confidence[i] > det_res.confidence[j]:
                                idx_to_skip.append(j)
                            else:
                                idx_to_skip.append(i)
            
            det_predictions[img_file] = []
            for detection_idx in range(len(det_res)):
                if detection_idx in idx_to_skip:
                    continue
                else:
                    cid = int(det_res.class_id[detection_idx])
                    name = PAI_mapping[cid]
                    xyxy = det_res.xyxy[detection_idx].astype(int)
                    det_predictions[img_file].append({
                        'class_id': cid, 
                        'class_name': name, 
                        'box': xyxy,
                        'confidence': float(det_res.confidence[detection_idx])
                    })
    
    # Print how many PAI masks were found
    print("Found {} PAI masks in {}.".format(len(det_predictions.get(img_file, [])), img_file))

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
