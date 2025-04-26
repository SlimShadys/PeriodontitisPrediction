import os

from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon
from tqdm import tqdm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box as shapely_box

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

tooth_colors, tooth_numbers_map = generate_tooth_colors(33)

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
SEG_MODEL_PATH = os.path.join(os.getcwd(), "yolo_seg", "best-dulcet-wildflower-40.pt")
DET_MODEL_PATH = os.path.join(os.getcwd(), "yolo_det", "best-lively-durian-49.pt")
SEG_CONF = 0.50 # Confidence threshold for segmentation
DET_CONF = 0.25 # Confidence threshold for detection
IMG_SZ = 1280   # Image size for YOLO model
DEVICE = "cuda" # Device to use for inference (e.g., "cuda" or "cpu")
AREA_OVERLAP_THRESHOLD = 0.05  # how much of the tooth‐mask must be covered to call it “affected”? e.g. 5% of the mask area

# Load models
yolo_seg = YOLO(SEG_MODEL_PATH)
yolo_det = YOLO(DET_MODEL_PATH)

DATA_DIR = os.path.join(os.getcwd(), "data", "InferenceData")
FINAL_DATA_DIR = os.path.join(DATA_DIR, "final")
if not os.path.exists(FINAL_DATA_DIR):
    os.makedirs(FINAL_DATA_DIR)
    
mask_predictions = {}
det_predictions = {}

# --- Inference Loop ---
files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.png'))] # Filter only .jpg and .png files
files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort files by number

for img_file in tqdm(os.listdir(DATA_DIR), desc="Processing images"):
    if not img_file.lower().endswith(('.jpg', '.png')):
        continue
    path = os.path.join(DATA_DIR, img_file)
    # ===== Segmentation
    seg_res = yolo_seg.predict(path, imgsz=IMG_SZ, conf=SEG_CONF, device=DEVICE, retina_masks=True, verbose=False)
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
    det_res = yolo_det.predict(path, imgsz=IMG_SZ, conf=DET_CONF, device=DEVICE, verbose=False)
    det_boxes = det_res[0].boxes
    if det_boxes:
        det_predictions[img_file] = []
        for box in det_boxes:
            cid = int(box.cls.cpu().numpy()[0])
            name = det_res[0].names[cid]
            xy = box.xyxy.cpu().squeeze().numpy().astype(int).tolist()
            det_predictions[img_file].append({'class_id': cid, 'class_name': name, 'box': xy})

# --- Match and Visualize ---
# Before drawing, make sure mask_prediction is sorted by the name of the file
mask_predictions = dict(sorted(mask_predictions.items(), key=lambda x: int(x[0].split('_')[1].split('.')[0])))

for img_file, masks in mask_predictions.items():
    detections = det_predictions.get(img_file, [])
    
    if not detections: # No detections for this image
        continue
    
    # … inside your per‐image loop, replacing the old compute_iou logic …
    matches = []
    for m in masks:
        mask_poly = Polygon(m['mask'])
        mask_area = mask_poly.area
        
        if mask_area == 0:
            continue

        for d in detections:
            x1, y1, x2, y2 = d['box']
            box_poly = shapely_box(x1, y1, x2, y2)

            # true overlapping area between mask‐polygon and disease‐box
            inter_area = mask_poly.intersection(box_poly).area

            # fraction of tooth mask area that’s “infected”
            frac = inter_area / mask_area

            if frac >= AREA_OVERLAP_THRESHOLD:
                matches.append({
                    'mask':       m['mask'],
                    'tooth':      tooth_numbers_map[m['class_id']],
                    'box':        d['box'],
                    'disease':    d['class_name'],
                    'overlap':    frac,
                    'mask_color': tooth_colors[m['class_id']],
                })
    
    # Only draw and save the object if matches are found
    if not matches:
        print(f"No matches found for {img_file}.")
        continue
    
    # Get drawing objects
    img = Image.open(os.path.join(DATA_DIR, img_file)).convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0,0,0,0))
    draw_m = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except:
        font = ImageFont.load_default()
    
    for item in matches:
        # Draw mask
        coords = [(int(x), int(y)) for x, y in item['mask']]
        col = item['mask_color'] + (128,)
        draw_m.polygon(coords, fill=col, outline=item['mask_color'] + (255,))
        
        # Draw box
        x1,y1,x2,y2 = item['box']
        draw.rectangle([x1,y1,x2,y2], outline=item['mask_color'], width=2)
        
        # Draw label
        text = f"{item['tooth']} - {item['disease']} ({item['overlap']:.2f})"
        
        # Draw text with black fill color
        draw.text((x1, y1-20), text, fill=(0, 0, 0, 128), font=font)
        
    # Combine and save
    out = Image.alpha_composite(img, overlay).convert('RGB')
    save_path = os.path.join(FINAL_DATA_DIR, img_file.replace('.', '_final.'))
    out.save(save_path)
    print(f"Saved: {save_path}")
