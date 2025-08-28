import os
import sys
from typing import Dict, Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import wandb.wandb_run
from torch.optim.lr_scheduler import (CosineAnnealingLR, LinearLR,
                                      SequentialLR, MultiStepLR)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
sys.path.append("./")
from misc.utils import get_dataset
from u_net.dataset import SegmentationDataset
from u_net.losses import CombinedLoss, DiceLoss, FocalLoss
from u_net.metrics import SegmentationMetrics
from u_net.model import UNet
from u_net.model2 import UNet2

class UNetTrainer:
    def __init__(self, model_configs: Dict, dataset_configs: Dict, wandb_run: Optional[wandb.wandb_run.Run]):
        self.model_configs = model_configs
        self.dataset_configs = dataset_configs
        self.wandb_run = wandb_run
        self.device = torch.device(model_configs['device'])
        self.ignore_index = 255  # Assuming 255 is the ignore index for segmentation masks

        if model_configs['model_type'] == 'unet++':
            self.model = smp.UnetPlusPlus(
                in_channels=1,
                classes=model_configs['num_classes'],
                encoder_name="resnet34",
                decoder_attention_type=None,
            ).to(self.device)
        else:
            self.model = UNet(
                n_channels=model_configs['num_channels'],
                n_classes=model_configs['num_classes'],
                bilinear=model_configs['bilinear'],
                use_se_enc=model_configs['use_se_enc'],
                use_se_dec=model_configs['use_se_dec'],
                use_aspp_block=model_configs['use_aspp_block'],
                use_dropout=model_configs['use_dropout'],
                use_residuals=model_configs['use_residuals'],
            ).to(self.device)

            # self.model = UNet2(
            #     n_channels=model_configs['num_channels'],
            #     n_classes=model_configs['num_classes'],
            #     bilinear=model_configs['bilinear'],
            #     use_aspp_block=model_configs['use_aspp_block'],
            #     use_residuals=model_configs['use_residuals'],
            # ).to(self.device)

        # Print model statistics
        print("Successfully initialized model {}.".format(self.model.__class__.__name__))
        print("Total parameters: {}".format(sum(p.numel() for p in self.model.parameters())))

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
        
        # Initialize warmup
        if model_configs['use_warmup']:
            warmup_scheduler = LinearLR(
                self.optimizer, start_factor=0.1, total_iters=model_configs['warmup_epochs']
            )

        # Initialize scheduler
        if model_configs['scheduler'] == 'cosine':
            self.lr_schedule = CosineAnnealingLR(
                self.optimizer, T_max=model_configs['epochs']
            )
        elif model_configs['scheduler'] == 'multistep':
            self.lr_schedule = MultiStepLR(
                self.optimizer, milestones=model_configs['milestones'], gamma=0.1
            )
        else:
            self.lr_schedule = None
        
        if model_configs['use_warmup'] and self.lr_schedule:
            self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, self.lr_schedule], milestones=[model_configs['warmup_epochs']])
        else:
            self.scheduler = self.lr_schedule

        # Initialize metrics
        self.train_metrics = SegmentationMetrics(model_configs['num_classes'])
        self.val_metrics = SegmentationMetrics(model_configs['num_classes'])
        
        # Best model tracking
        self.best_fitness = 0.0
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
            
            # Calculate fitness
            fitness = 0.45 * val_metrics['mIoU'] + 0.45 * val_metrics['mDice'] + 0.10 * (1 - val_loss)

            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train mIoU: {train_metrics['mIoU']:.4f}, Val mIoU: {val_metrics['mIoU']:.4f}")
            print(f"Train mDice: {train_metrics['mDice']:.4f}, Val mDice: {val_metrics['mDice']:.4f}")
            print(f"Fitness: {fitness:.4f}")
            #print(f"Train mAP50: {train_metrics['mAP50']:.4f}, Val mAP50: {val_metrics['mAP50']:.4f}")
            #print(f"Train mAP50-95: {train_metrics['mAP50-95']:.4f}, Val mAP50-95: {val_metrics['mAP50-95']:.4f}")
                       
            # Save best model
            if fitness > self.best_fitness:
                self.best_fitness = fitness
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
                print(f"New best model saved with the following fitness: {self.best_fitness:.4f}")
            else:
                self.patience_counter += 1
                
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
                    'fitness': fitness,
                    # 'train_mAP50': train_metrics['mAP50'],
                    # 'val_mAP50': val_metrics['mAP50'],
                    # 'train_mAP50-95': train_metrics['mAP50-95'],
                    # 'val_mAP50-95': val_metrics['mAP50-95'],
                    'lr': self.optimizer.param_groups[0]['lr']
                })
       
            # Early stopping
            if self.patience_counter >= self.model_configs['patience']:
                print(f"Early stopping triggered after {self.patience_counter} epochs without any improvement. :(")
                break

def train_model(model_configs: Dict, dataset_configs: Dict, wandb_run: Optional[wandb.wandb_run.Run]) -> None:
    """Main training function"""
    
    # Create unified dataset
    full_dataset = SegmentationDataset(
        data_path=dataset_configs['path'],
        img_size=model_configs['imgsz'],
        val_size=dataset_configs['val_size'] if 'val_size' in dataset_configs else 0.2,
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
    use_wandb = False
    
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
        'val_size': 0.2,
        'create_yolo_version': False,
        'enhance_images': False,
    }

    model_configs = {
        'model_type': 'unet',   # [unet++, unet]
        'num_channels': 1,      # Number of input channels (3 for RGB, 1 for Grayscale)
        'num_classes': 33,      # Number of output classes
        'bilinear': False,      # Use bilinear upsampling in the Decoder
        'use_residuals': True,  # Use residual connections for Encoder/Decoder
        'use_se_enc': False,    # Use Attention Modules in the Encoder
        'use_se_dec': False,    # Use Attention Modules in the Decoder
        'use_aspp_block': True, # Use ASPP block for Bottleneck and final Layer
        'use_dropout': False,   # Use Dropout at final Layer
        'device': f'cuda:{CUDA_DEVICE}' if isinstance(CUDA_DEVICE, int) else ','.join(map(str, CUDA_DEVICE)),

        # Training parameters
        'imgsz': 512,
        'epochs': 200,
        'batch': 2,
        'workers': 0,
        'optimizer': 'SGD',  # [Adam, AdamW, SGD]
        'lr0': 5e-3,
        'momentum': 0.99,
        'weight_decay': 1e-4,
        'scheduler': 'cosine',  # [cosine, multistep, none]
        'patience': 35,
        'use_warmup': True,
        'warmup_epochs': 5,  # Single integer for warmup duration
        'milestones': [50, 100, 150],  # List for SequentialLR milestones
        
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
