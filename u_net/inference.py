import importlib
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Local imports
sys.path.append("./")
from u_net.model import UNet

class UNetInference:
    def __init__(self, model_path: str, device: str = 'cuda:0'):
        """
        Initialize UNet inference
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model_configs = checkpoint['model_configs']
        
        # Initialize model
        self.model = UNet(
            n_channels=3, 
            n_classes=self.model_configs['num_classes']
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Image size for inference
        self.img_size = self.model_configs['imgsz']
        
        # Create color map for visualization
        self.color_map = self._create_color_map()
        
        print(f"Model loaded successfully!")
        print(f"Number of classes: {self.model_configs['num_classes']}")
        print(f"Model performance - mIoU: {checkpoint.get('val_miou', 'N/A'):.4f}")
        print(f"Device: {self.device}")

    def evaluate_map_on_validation_set(self, dataset_module_path: str, dataset_class_name: str, dataset_args: dict, batch_size: int = 1, num_workers: int = 0):
        """
        Evaluate mAP on the validation set using the dataset split logic from train.py.
        Args:
            dataset_module_path: Python path to the dataset module (e.g. 'u_net.dataset')
            dataset_class_name: Name of the dataset class (e.g. 'SegmentationDataset')
            dataset_args: Arguments to instantiate the dataset (dict)
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
        """
        # Dynamically import the dataset class
        dataset_module = importlib.import_module(dataset_module_path)
        DatasetClass = getattr(dataset_module, dataset_class_name)
        full_dataset = DatasetClass(**dataset_args)
        val_dataset = full_dataset.get_val_dataset(augment=False)
        from torch.utils.data import DataLoader
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        pred_masks = []
        gt_masks = []
        print(f"Evaluating mAP on validation set: {len(val_dataset)} samples...")
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation mAP"):
                images = images.to(self.device)
                outputs = self.model(images)
                preds = torch.argmax(F.softmax(outputs, dim=1), dim=1).cpu().numpy()
                masks = masks.cpu().numpy()
                for p, g in zip(preds, masks):
                    pred_masks.append(p)
                    gt_masks.append(g)
        results = self.compute_map(pred_masks, gt_masks)
        print(f"Validation mAP@0.5: {results['mAP@0.5']:.4f}")
        print(f"Validation mAP@0.5:0.95: {results['mAP@0.5:0.95']:.4f}")
        return results
    
    def extract_instances(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract connected components (instances) from a mask.
        Returns a mask where each instance has a unique label (0 is background).
        """
        from skimage import measure
        return measure.label(mask, background=0)

    def compute_iou_matrix(self, pred_instances: np.ndarray, gt_instances: np.ndarray) -> np.ndarray:
        """
        Compute IoU matrix between predicted and ground truth instances.
        Returns IoU matrix, pred_ids, gt_ids.
        """
        pred_ids = np.unique(pred_instances)
        pred_ids = pred_ids[pred_ids != 0]  # skip background
        gt_ids = np.unique(gt_instances)
        gt_ids = gt_ids[gt_ids != 0]
        iou_matrix = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
        for i, p in enumerate(pred_ids):
            pred_mask = pred_instances == p
            for j, g in enumerate(gt_ids):
                gt_mask = gt_instances == g
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                iou_matrix[i, j] = intersection / union if union > 0 else 0.0
        return iou_matrix, pred_ids, gt_ids

    def compute_map(self, pred_masks: list, gt_masks: list, iou_thresholds=None) -> dict:
        """
        Compute mAP@0.5 and mAP@0.5:0.95 for a list of predicted and ground truth masks.
        Each mask should be a 2D numpy array (H, W) with instance labels.
        """
        if iou_thresholds is None:
            iou_thresholds = np.arange(0.5, 1.0, 0.05)
        aps = []
        ap50 = []
        for t in iou_thresholds:
            tp, fp, fn = 0, 0, 0
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                pred_instances = self.extract_instances(pred_mask)
                gt_instances = self.extract_instances(gt_mask)
                iou_matrix, pred_ids, gt_ids = self.compute_iou_matrix(pred_instances, gt_instances)
                matched_gt = set()
                matched_pred = set()
                # Match predictions to ground truth
                for i, p in enumerate(pred_ids):
                    for j, g in enumerate(gt_ids):
                        if iou_matrix[i, j] >= t and j not in matched_gt and i not in matched_pred:
                            tp += 1
                            matched_gt.add(j)
                            matched_pred.add(i)
                fp += len(pred_ids) - len(matched_pred)
                fn += len(gt_ids) - len(matched_gt)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            ap = precision  # For instance segmentation, AP is precision at this threshold
            aps.append(ap)
            if abs(t - 0.5) < 1e-6:
                ap50.append(ap)
        map_50_95 = np.mean(aps)
        map_50 = np.mean(ap50)
        return {"mAP@0.5": map_50, "mAP@0.5:0.95": map_50_95}

    def _create_color_map(self) -> np.ndarray:
        """Create color map for visualization"""
        # Generate distinct colors for each class
        colors = []
        for i in range(self.model_configs['num_classes']):
            if i == 0:  # Background class
                colors.append([0, 0, 0])  # Black background
            else:
                # Create distinct colors using HSV space for non-background classes
                hue = ((i-1) * 137.5) % 360  # Golden angle approximation, adjusted for i-1
                saturation = 0.8
                value = 0.9
                
                # Convert HSV to RGB
                c = value * saturation
                x = c * (1 - abs(((hue / 60) % 2) - 1))
                m = value - c
                
                if 0 <= hue < 60:
                    r, g, b = c, x, 0
                elif 60 <= hue < 120:
                    r, g, b = x, c, 0
                elif 120 <= hue < 180:
                    r, g, b = 0, c, x
                elif 180 <= hue < 240:
                    r, g, b = 0, x, c
                elif 240 <= hue < 300:
                    r, g, b = x, 0, c
                else:
                    r, g, b = c, 0, x
                
                colors.append([(r + m) * 255, (g + m) * 255, (b + m) * 255])
        
        return np.array(colors, dtype=np.uint8)
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed tensor and original image
        """
        # Load image
        if isinstance(image_path, str):
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            original_image = image_path
            
        # Resize image
        resized_image = cv2.resize(original_image, (self.img_size, self.img_size))
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(resized_image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        
        return image_tensor.to(self.device), original_image
    
    def postprocess_output(self, output: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess model output
        
        Args:
            output: Model output tensor
            original_shape: Original image shape (H, W)
            
        Returns:
            Segmentation mask
        """
        # Apply softmax and get predictions
        probs = F.softmax(output, dim=1)
        predictions = torch.argmax(probs, dim=1)
        
        # Convert to numpy
        mask = predictions.cpu().numpy()[0]  # Remove batch dimension
        
        # Resize to original shape
        mask = cv2.resize(mask.astype(np.uint8), (original_shape[1], original_shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def create_colored_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Create colored visualization of segmentation mask
        
        Args:
            mask: Segmentation mask with class indices
            
        Returns:
            Colored mask
        """
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        for class_id in range(self.model_configs['num_classes']):
            colored_mask[mask == class_id] = self.color_map[class_id]
        
        return colored_mask
    
    def create_overlay(self, original_image: np.ndarray, mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """
        Create overlay of original image and segmentation mask
        
        Args:
            original_image: Original image
            mask: Segmentation mask
            alpha: Transparency factor
            
        Returns:
            Overlaid image
        """
        colored_mask = self.create_colored_mask(mask)
        overlay = cv2.addWeighted(original_image, alpha, colored_mask, 1 - alpha, 0)
        return overlay
    
    def predict_single_image(self, image_path: str, save_results: bool = True, 
                           output_dir: str = "inference_results") -> Dict:
        """
        Run inference on a single image
        
        Args:
            image_path: Path to input image
            save_results: Whether to save results
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing results
        """
        # Preprocess image
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor)
        
        # Postprocess output
        mask = self.postprocess_output(output, original_image.shape[:2])
        
        # Create visualizations
        colored_mask = self.create_colored_mask(mask)
        overlay = self.create_overlay(original_image, mask)
        
        # Save results if requested
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = Path(image_path).stem
            
            # Save original image
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.jpg"), 
                       cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
            
            # Save colored mask
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.jpg"), 
                       cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
            
            # Save overlay
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_overlay.jpg"), 
                       cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            # Save raw mask
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_raw_mask.png"), mask)
        
        # Calculate class statistics
        unique_classes, counts = np.unique(mask, return_counts=True)
        class_stats = {int(cls): int(count) for cls, count in zip(unique_classes, counts)}
        
        return {
            'mask': mask,
            'colored_mask': colored_mask,
            'overlay': overlay,
            'class_stats': class_stats,
            'original_image': original_image
        }
    
    def predict_batch(self, image_paths: List[str], output_dir: str = "inference_results") -> List[Dict]:
        """
        Run inference on batch of images
        
        Args:
            image_paths: List of image paths
            output_dir: Directory to save results
            
        Returns:
            List of results for each image
        """
        results = []
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                result = self.predict_single_image(image_path, save_results=True, output_dir=output_dir)
                results.append(result)
                print(f"Processed: {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        return results
    
    def predict_directory(self, input_dir: str, output_dir: str = "inference_results", 
                         extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']) -> List[Dict]:
        """
        Run inference on all images in a directory
        
        Args:
            input_dir: Input directory containing images
            output_dir: Directory to save results
            extensions: Supported image extensions
            
        Returns:
            List of results for each image
        """
        # Find all image files
        image_paths = []
        for ext in extensions:
            image_paths.extend(Path(input_dir).glob(f"*{ext}"))
            image_paths.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        image_paths = [str(p) for p in image_paths]
        
        print(f"Found {len(image_paths)} images in {input_dir}")
        
        return self.predict_batch(image_paths, output_dir)
    
    def create_summary_report(self, results: List[Dict], output_dir: str = "inference_results"):
        """
        Create summary report of inference results
        
        Args:
            results: List of inference results
            output_dir: Directory to save report
        """
        # Calculate overall statistics
        total_pixels = sum(sum(result['class_stats'].values()) for result in results)
        overall_class_stats = {}
        
        for result in results:
            for class_id, count in result['class_stats'].items():
                overall_class_stats[class_id] = overall_class_stats.get(class_id, 0) + count
        
        # Create report
        report = {
            'total_images': len(results),
            'total_pixels': total_pixels,
            'class_distribution': {
                class_id: {
                    'pixel_count': count,
                    'percentage': (count / total_pixels) * 100
                }
                for class_id, count in overall_class_stats.items()
            }
        }
        
        # Save report
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'inference_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Summary report saved to {output_dir}/inference_report.json")
        
        # Print summary
        print("\n" + "="*50)
        print("INFERENCE SUMMARY")
        print("="*50)
        print(f"Total images processed: {len(results)}")
        print(f"Total pixels: {total_pixels:,}")
        print("\nClass distribution:")
        for class_id, stats in report['class_distribution'].items():
            print(f"  Class {class_id}: {stats['pixel_count']:,} pixels ({stats['percentage']:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='UNet Inference Script')
    parser.add_argument('--model_path', type=str, default='runs/unet/v0.1/dainty-cosmos-32/best_model.pth', 
                        help='Path to trained model checkpoint (default: runs/unet/v0.1/dainty-cosmos-32/best_model.pth)')
    parser.add_argument('--input', type=str, default='data/InferenceData', 
                        help='Input image path or directory (default: data/InferenceData)')
    parser.add_argument('--output', type=str, default='data/InferenceData/UNet_results',
                        help='Output directory for results (default: data/InferenceData/UNet_results)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run inference on (default: cuda:0)')
    parser.add_argument('--batch', action='store_true', default=True,
                        help='Process directory of images (default: True)')
    parser.add_argument('--evaluate_map', action='store_true', default=False,
                        help='Evaluate mAP on validation set (default: False)')
    parser.add_argument('--dataset_module', type=str, default='u_net.dataset', help='Module path for dataset (default: u_net.dataset)')
    parser.add_argument('--dataset_class', type=str, default='SegmentationDataset', help='Dataset class name (default: SegmentationDataset)')
    parser.add_argument('--dataset_args', type=str, default=None, help='JSON string of dataset args (e.g. {"data_path": "...", "img_size": 512})')
    
    # Parse arguments    
    args = parser.parse_args()
    
    # If no arguments are provided (run from VS Code play button), use sensible defaults for mAP on val set
    if args.evaluate_map:
        # You can edit these defaults as needed:
        default_model_path = 'runs/unet/v0.1/dainty-cosmos-32/best_model.pth'
        default_dataset_args = {
            'data_path': os.path.join(os.getcwd(), "data", "DualLabel"), # For local testing
            # 'data_path': os.path.abspath(os.path.join(os.getcwd(), "..", "datasets", "DualLabel")), # For Docker testing
            "img_size": 512,
            "val_size": 0.2,
            "random_state": 2047315
        }
        inference = UNetInference(default_model_path, 'cuda:0')
        inference.evaluate_map_on_validation_set(
            dataset_module_path='u_net.dataset',
            dataset_class_name='SegmentationDataset',
            dataset_args=default_dataset_args,
        )
        return

    # Initialize inference
    inference = UNetInference(args.model_path, args.device)

    # Run inference
    if args.batch or os.path.isdir(args.input):
        print(f"Processing directory: {args.input}")
        results = inference.predict_directory(args.input, args.output)
        inference.create_summary_report(results, args.output)
    else:
        print(f"Processing single image: {args.input}")
        result = inference.predict_single_image(args.input, save_results=True, output_dir=args.output)
        print(f"Results saved to: {args.output}")
        # Print class statistics
        print("\nClass statistics:")
        for class_id, count in result['class_stats'].items():
            print(f"  Class {class_id}: {count} pixels")

if __name__ == '__main__':
    main()
