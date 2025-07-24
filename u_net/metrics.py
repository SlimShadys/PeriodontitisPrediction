import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from tqdm import tqdm

class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        # Store per-class predictions more efficiently
        self.class_predictions = [[] for _ in range(self.num_classes)]
        self.class_targets = [[] for _ in range(self.num_classes)]

    def update(self, preds, targets):
        """
        preds: (N, C, H, W) logits
        targets: (N, H, W) class indices
        """
        preds = F.softmax(preds, dim=1)
        pred_classes = torch.argmax(preds, dim=1)
        
        # Flatten
        pred_classes = pred_classes.cpu().numpy().flatten()
        targets_flat = targets.cpu().numpy().flatten()
        
        # Remove ignore_index
        mask = targets_flat != self.ignore_index
        pred_classes = pred_classes[mask]
        targets_flat = targets_flat[mask]
        
        # Update confusion matrix
        for t, p in zip(targets_flat, pred_classes):
            self.confusion_matrix[t, p] += 1
        
        # # NEEDED FOR MAP CALCULATION
        # # Store predictions more efficiently - only store what we need
        # preds_np = preds.detach().cpu().numpy()
        # targets_np = targets.cpu().numpy()
        
        # for class_idx in range(self.num_classes):
        #     # Get binary targets for this class
        #     class_targets = (targets_np == class_idx).astype(int).flatten()
        #     # Get prediction probabilities for this class
        #     class_preds = preds_np[:, class_idx].flatten()
            
        #     # Only store if there are any positive samples
        #     if class_targets.sum() > 0:
        #         self.class_predictions[class_idx].append(class_preds)
        #         self.class_targets[class_idx].append(class_targets)

    def compute_iou_per_class(self):
        ious = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            if tp + fp + fn == 0:
                ious.append(float('nan'))
            else:
                iou = tp / (tp + fp + fn)
                ious.append(iou)
        return np.array(ious)

    def compute_dice_per_class(self):
        dice_scores = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            if 2*tp + fp + fn == 0:
                dice_scores.append(float('nan'))
            else:
                dice = (2 * tp) / (2*tp + fp + fn)
                dice_scores.append(dice)
        return np.array(dice_scores)

    def compute_map(self):
        """Compute mAP - simplified for segmentation"""
        if not any(self.class_predictions):
            return 0.0, 0.0
        
        aps = []
        
        for class_idx in tqdm(range(self.num_classes), desc="Computing mAP"):
            if not self.class_predictions[class_idx]:
                continue
                
            # Concatenate predictions and targets for this class only
            class_preds = np.concatenate(self.class_predictions[class_idx])
            class_targets = np.concatenate(self.class_targets[class_idx])
            
            if class_targets.sum() == 0:
                continue
            
            # Compute AP for this class
            try:
                ap = average_precision_score(class_targets, class_preds)
                aps.append(ap)
            except:
                continue
        
        map_score = np.mean(aps) if aps else 0.0
        return map_score, map_score  # For segmentation, mAP50 ≈ mAP50-95

    def get_metrics(self):
        ious = self.compute_iou_per_class()
        dice_scores = self.compute_dice_per_class()
        #map_50, map_50_95 = self.compute_map()
        
        # Remove NaN values for mean calculation
        valid_ious = ious[~np.isnan(ious)]
        valid_dice = dice_scores[~np.isnan(dice_scores)]
        
        return {
            'mIoU': np.mean(valid_ious) if len(valid_ious) > 0 else 0.0,
            'mDice': np.mean(valid_dice) if len(valid_dice) > 0 else 0.0,
            #'mAP50': map_50,
            #'mAP50-95': map_50_95,
            'IoU_per_class': ious,
            'Dice_per_class': dice_scores
        }
