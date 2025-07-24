import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)
        num_classes = preds.shape[1]

        one_hot_targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1)
            preds = preds * mask
            one_hot_targets = one_hot_targets * mask

        intersection = (preds * one_hot_targets).sum(dim=(0, 2, 3))
        union = preds.sum(dim=(0, 2, 3)) + one_hot_targets.sum(dim=(0, 2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5, ignore_index=255):
        super(CombinedLoss, self).__init__()
        
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.dice_loss = DiceLoss(ignore_index=self.ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    
    def forward(self, preds, targets):
        ce = self.ce_loss(preds, targets)
        dice = self.dice_loss(preds, targets)
        return self.ce_weight * ce + self.dice_weight * dice
