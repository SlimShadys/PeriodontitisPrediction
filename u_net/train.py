import os
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import wandb.wandb_run
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
sys.path.append("./")
from misc.utils import get_dataset
from u_net.dataset import SegmentationDataset
from u_net.losses import CombinedLoss, DiceLoss, FocalLoss
from u_net.metrics import SegmentationMetrics
from u_net.model import UNet

class UNetTrainer:
    def __init__(self, model_configs: Dict, dataset_configs: Dict, wandb_run: Optional[wandb.wandb_run.Run]):
        self.model_configs = model_configs
        self.dataset_configs = dataset_configs
        self.wandb_run = wandb_run
        self.device = torch.device(model_configs['device'])
        self.ignore_index = 255  # Assuming 255 is the ignore index for segmentation masks
        
        # Initialize model
        self.model = UNet(
            n_channels=3, 
            n_classes=model_configs['num_classes'],
        ).to(self.device)
        
        # Initialize loss function
        if model_configs['loss_type'] == 'combined':
            self.criterion = CombinedLoss(
                ce_weight=model_configs['ce_weight'],
                dice_weight=model_configs['dice_weight'],
                ignore_index=self.ignore_index
            )
        elif model_configs['loss_type'] == 'dice':
            self.criterion = DiceLoss(
                ignore_index=self.ignore_index
            )
        elif model_configs['loss_type'] == 'focal':
            self.criterion = FocalLoss(
                alpha=model_configs['focal_alpha'],
                gamma=model_configs['focal_gamma'],
                ignore_index=self.ignore_index
            )
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        
        # Initialize optimizer
        if model_configs['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=model_configs['lr0'],
                weight_decay=model_configs['weight_decay']
            )
        elif model_configs['optimizer'] == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=model_configs['lr0'],
                weight_decay=model_configs['weight_decay']
            )
        else:  # SGD
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=model_configs['lr0'],
                momentum=model_configs['momentum'],
                weight_decay=model_configs['weight_decay']
            )
        
        # Initialize scheduler
        if model_configs['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=model_configs['epochs']
            )
        elif model_configs['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        else:
            self.scheduler = None
        
        # Initialize metrics
        self.train_metrics = SegmentationMetrics(model_configs['num_classes'])
        self.val_metrics = SegmentationMetrics(model_configs['num_classes'])
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_miou = 0.0
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        self.model.train()
        self.train_metrics.reset()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            self.train_metrics.update(outputs, masks)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return running_loss / len(train_loader)

    def validate_epoch(self, val_loader):
        self.model.eval()
        self.val_metrics.reset()
        running_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                
                loss = self.criterion(outputs, masks)
                
                running_loss += loss.item()
                self.val_metrics.update(outputs, masks)

                pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        return running_loss / len(val_loader)

    def train(self, train_loader, val_loader):
        for epoch in range(self.model_configs['epochs']):
            print(f"\nEpoch {epoch+1}/{self.model_configs['epochs']}")
            
            # Training phase
            train_loss = self.train_epoch(train_loader)
            train_metrics = self.train_metrics.get_metrics()
            
            # Validation phase
            val_loss = self.validate_epoch(val_loader)
            val_metrics = self.val_metrics.get_metrics()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train mIoU: {train_metrics['mIoU']:.4f}, Val mIoU: {val_metrics['mIoU']:.4f}")
            #print(f"Train mAP50: {train_metrics['mAP50']:.4f}, Val mAP50: {val_metrics['mAP50']:.4f}")
            #print(f"Train mAP50-95: {train_metrics['mAP50-95']:.4f}, Val mAP50-95: {val_metrics['mAP50-95']:.4f}")
            
            # Log to wandb
            if self.wandb_run:
                self.wandb_run.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_mIoU': train_metrics['mIoU'],
                    'val_mIoU': val_metrics['mIoU'],
                    'train_mDice': train_metrics['mDice'],
                    'val_mDice': val_metrics['mDice'],
                    # 'train_mAP50': train_metrics['mAP50'],
                    # 'val_mAP50': val_metrics['mAP50'],
                    # 'train_mAP50-95': train_metrics['mAP50-95'],
                    # 'val_mAP50-95': val_metrics['mAP50-95'],
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_metrics['mIoU'] > self.best_miou:
                self.best_miou = val_metrics['mIoU']
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save model
                save_path = os.path.join(self.model_configs['save_dir'], 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_miou': val_metrics['mIoU'],
                    'model_configs': self.model_configs
                }, save_path)
                print(f"New best model saved with mIoU: {self.best_miou:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.model_configs['patience']:
                print(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                break

def train_model(model_configs: Dict, dataset_configs: Dict, wandb_run: Optional[wandb.wandb_run.Run]) -> None:
    """Main training function"""
    
    # Create unified dataset
    full_dataset = SegmentationDataset(
        data_path=dataset_configs['path'],
        img_size=model_configs['imgsz'],
        val_size=model_configs['val_size'] if 'val_size' in dataset_configs else 0.2,
    )

    # Get train and val splits
    train_dataset = full_dataset.get_train_dataset(augment=model_configs['augment'])
    val_dataset = full_dataset.get_val_dataset(augment=False)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_configs['batch'],
        shuffle=True,
        num_workers=model_configs['workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=model_configs['batch'],
        shuffle=False,
        num_workers=model_configs['workers'],
        pin_memory=True
    )
    
    # Update wandb config
    if wandb_run:
        wandb_run.config.update(model_configs)
        wandb_run.config.update(dataset_configs)
    
    # Initialize trainer and start training
    trainer = UNetTrainer(model_configs, dataset_configs, wandb_run)
    trainer.train(train_loader, val_loader)

def main():
    # ============== PARAMETERS ============== #
    
    # WandB configs
    use_wandb = True
    
    if use_wandb:
        wandb.login()
        entity = "SlimShadys"
        project = "FIS2-UNetSegmentation"
        group = "v0.1"
        wandb_id = None
    else:
        version = "v0.1"

    dataset_configs = {
        'task_type': 'segmentation',
        'name': 'DualLabel',
        'path': os.path.join(os.getcwd(), "data", "DualLabel"), # For local testing
        # 'path': os.path.abspath(os.path.join(os.getcwd(), "..", "datasets", "DualLabel")), # For Docker testing
        'create_yolo_version': False,
        'enhance_images': False,
    }

    model_configs = {
        'model_type': 'unet',
        'num_classes': 33,
        'device': f'cuda:{CUDA_DEVICE}' if isinstance(CUDA_DEVICE, int) else ','.join(map(str, CUDA_DEVICE)),
        
        # Training parameters
        'imgsz': 512,
        'epochs': 100,
        'batch': 4,
        'workers': 0,
        'optimizer': 'AdamW',  # [Adam, AdamW, SGD]
        'lr0': 1e-4,
        'momentum': 0.937,
        'weight_decay': 5e-4,
        'scheduler': 'cosine',  # [cosine, step, none]
        'patience': 20,
        
        # Loss configuration
        'loss_type': 'combined',  # [combined, dice, focal, ce]
        'ce_weight': 0.5,
        'dice_weight': 0.5,
        'focal_alpha': 1.0,
        'focal_gamma': 2.0,
        
        # Augmentation
        'augment': True,
   
        # Resume option
        'resume': False,
    }

    # Get dataset
    dataset_configs['data'] = get_dataset(dataset_configs)

    # Init WandB
    if use_wandb:
        if model_configs["resume"] and wandb_id:
            resume = "allow"
            id = wandb_id
        else:
            resume = None
            id = None

        wandb_run = wandb.init(
            entity=entity,
            project=project,
            group=group,
            resume=resume,
            id=id
        )
        model_configs["save_dir"] = os.path.join(os.getcwd(), "runs", "unet", group, wandb_run.name)
    else:
        wandb_run = None
        model_configs["save_dir"] = os.path.join(os.getcwd(), "runs", "unet", version)

    # Create save directory
    os.makedirs(model_configs["save_dir"], exist_ok=True)

    # Train the model
    train_model(model_configs, dataset_configs, wandb_run)

if __name__ == '__main__':
    CUDA_DEVICE = 0
    
    available_gpus = torch.cuda.device_count()
    print("Available GPUs:", available_gpus)
    for i in range(available_gpus):
        print(f"\t- GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if isinstance(CUDA_DEVICE, list):
        devices = ",".join(map(str, CUDA_DEVICE))
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        print(f"Setting CUDA devices with IDs: {devices}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
        torch.cuda.set_device(CUDA_DEVICE)
        print(f"Successfully set CUDA device with ID: {torch.cuda.current_device()}")
    
    main()
    print("\n\nUNet training completed successfully! 🎉")
