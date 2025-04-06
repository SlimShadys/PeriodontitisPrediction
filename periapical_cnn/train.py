import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.models.resnet import ResNet50_Weights
from tqdm import tqdm
import wandb
from early_stopping_pytorch import EarlyStopping

# Local imports
sys.path.append("./")
from misc import dataset_builder
from misc.scheduler import WarmupDecayLR
from misc.utils import save_model

# Set the random seed for reproducibility
torch.manual_seed(2047315)
np.random.seed(2047315)
random.seed(2047315)

def main():
    
    # WandB configs
    use_wandb = True                    # Set to True if you want to use WandB for logging | # If True, also run ```yolo settings wandb=True``` in the terminal
    if use_wandb:
        wandb.login()
        entity = "SlimShadys"           # Set the wandb entity where your project will be logged (generally your team name).
        project = "FIS2-CNNPeriodontitisDetection"  # Set the wandb project where this run will be logged.
        group = "v0.1"                  # Set the group name of the run, this is useful when comparing multiple runs in a project.
        wandb_id = None                 # Set the run ID if you want to resume a run (plus 'resume' must be set to True in the model_configs)
    else:
        version = "v0.1"                # Set the version of the run (saved locally ONLY)
    
    model_configs = {
        # == Dataset-related
        'data_path': os.path.abspath(os.path.join(os.getcwd(), "..", "datasets")), # Path to the dataset (for Docker)
        # 'data_path': 'data', # Path to the dataset (local testing)
        'dataset_name': 'periapical',
        # == Training-related
        'model_name': 'resnet50',
        'batch_size': 64, # Batch size for training
        'start_epoch': 0,
        'epochs': 20,          # Number of epochs to train
        'val_interval': 1,
        'log_interval': 50,     # Log interval for training
        'output_dir': os.path.abspath(os.path.join(
            os.getcwd(), "periapical_cnn", group if use_wandb else version, "output")), # Path to save the model checkpoints (for Docker)
        # 'output_dir': os.path.join("periapical_cnn", group if use_wandb else version, "output"), # Path to save the model checkpoints (local testing)
        # == Optimizer-related
        'optimizer': 'adam',    # Optimizer to use (adam, adamw, sgd)
        'learning_rate': 1.5e-4, # Learning rate for the optimizer
        # == Scheduler-related
        'use_warmup': True,     # Use warmup for the learning rate
        'warmup_epochs': 5,     # Number of epochs for warmup
        'steps': [3, 10],  # Epochs to decay the learning rate (ALWAYS active for now)
        'gamma': 0.1,           # Decay factor for the learning rate
        'decay_method': "cosine", # Decay method for the learning rate (linear, smooth, cosine)
        'cosine_power': 0.5,    # Power for the cosine decay
        'min_lr': 1.0e-6,       # Minimum learning rate
        # 'momentum': 0.937,
        'weight_decay': 5e-4,
        # == Network-related
        'image_size': 320, # Image size for resizing augmentation
        'patience': 7, # Patience for early stopping
        # == Resume option
        'resume': False, # Resume training from a checkpoint (for wandb)
    }

    # Augmentation configurations
    augmentation_configs = {
        'RESIZE': (model_configs['image_size'], model_configs['image_size']),
        'RANDOM_CROP': (0, 0),
        'RANDOM_HORIZONTAL_FLIP_PROB': 0.0,
        'RANDOM_ERASING_PROB': 0.0,
        'JITTER_BRIGHTNESS': 0.0,
        'JITTER_CONTRAST': 0.0,
        'JITTER_SATURATION': 0.0,
        'JITTER_HUE': 0.0,
        'COLOR_AUGMENTATION': False,
        'PADDING': 0,
        'NORMALIZE_MEAN': None,
        'NORMALIZE_STD': None
    }
    
    # Configure the output path (if it exists)
    if not os.path.exists(model_configs['output_dir']):
        os.makedirs(model_configs['output_dir'], exist_ok=True)

    # Define the Dataset
    dataset = dataset_builder.DatasetBuilder(data_path=model_configs['data_path'], dataset_name=model_configs['dataset_name'], augmentation_configs=augmentation_configs)

    # Define the DataLoaders
    train_loader = DataLoader(dataset.train_set, batch_size=model_configs['batch_size'], shuffle=True, collate_fn=dataset.train_set.train_collate_fn)
    validation_loader = DataLoader(dataset.validation_set, batch_size=model_configs['batch_size'], shuffle=False, collate_fn=dataset.validation_set.val_collate_fn)

    # Define the model
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 3) # 3 classes for classification: [PAI 3, PAI 4, PAI 5]
    model = model.cuda()

    # Move the model to GPUs if available
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=model_configs['learning_rate'], weight_decay=model_configs['weight_decay'])
    
    # Define the learning rate scheduler
    if model_configs['use_warmup'] == True:
        # For Learning Rate Warm-up and Decay
        # We first warm-up the LR up for a few epochs, so that it reaches the desired LR
        # Then, we decay the LR using the milestones or a smooth/cosine decay
        scheduler = WarmupDecayLR(
            optimizer=optimizer,
            milestones=model_configs['steps'],
            warmup_epochs=model_configs['warmup_epochs'],
            gamma=model_configs['gamma'],
            cosine_power=model_configs['cosine_power'],
            decay_method=model_configs['decay_method'],
            min_lr=model_configs['min_lr'],
        )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=model_configs['steps'], gamma=model_configs['gamma'])

    # Define the early stopping criteria if patience is a value > 0
    if model_configs['patience'] > 0:
        early_stopping = EarlyStopping(patience=model_configs['patience'], verbose=True)

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Init WandB
    if use_wandb:
        # Check if we are resuming a run
        if model_configs["resume"] == True and wandb_id is not None:
            print("Resuming a WandB run with ID:", wandb_id)
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
    else:
        wandb_run = None

    # Update the WandB configs before training
    if wandb_run is not None:
        wandb_run.config.update({"model_configs": model_configs})
        wandb_run.config.update({"augmentation_configs": augmentation_configs})

    # Vars
    running_loss, acc_list, val_running_loss, val_acc_list = [], [], [], []

    # Training loop
    for epoch in range(model_configs['start_epoch'], model_configs['epochs']):
        
        # Ensure the model is in training mode
        model.train()
        
        # Empty the log lists
        running_loss, acc_list, val_running_loss, val_acc_list = [], [], [], []
        
        # Start the epoch
        for batch_idx, (img_paths, imgs, widths, heights, depths, objects, infos) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{model_configs['epochs']}"):
            
            # Get the labels from the objects
            labels = [obj[0] for objs in objects for obj in objs]
            labels = torch.stack(labels, dim=0)
            labels = labels.cuda() # [batch_size, 1]
            
            # Move the images to the device
            if type(imgs) == list or type(imgs) == tuple:
                imgs = [im.cuda() for im in imgs]
            else:
                imgs = imgs.cuda()

            # Equivalent to: optimizer.zero_grad()
            # But explicitly setting the gradients to None for each parameter
            # for optimizing memory usage and speed.
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass on the Network
            outputs = model(imgs)
            pred_ids = torch.argmax(outputs, dim=1)  # Get predicted class IDs

            # Loss calculation
            loss = loss_fn(outputs, labels) # ID loss

            # Accuracy
            accuracy = (pred_ids == labels).float().mean()
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss.append(loss.detach().cpu().item())
            acc_list.append(accuracy.detach().cpu().item())

            if (batch_idx % model_configs['log_interval'] == 0):
                print(f"\tIteration[{batch_idx}/{len(train_loader)}] "
                        f"Loss: {np.mean(running_loss):.4f} | "
                        f"Accuracy: {np.mean(acc_list):.4f} | "
                        f"LR: {scheduler.optimizer.param_groups[0]['lr']:.2e}")

        if use_wandb:
            log_dict = {
                'epoch': epoch,
                'train_loss': np.mean(running_loss),
                'train_accuracy': np.mean(acc_list),
                'lr': optimizer.param_groups[0]['lr'],
            }

        # Update learning rate
        scheduler.step()

        # Log the current learning rate
        opt_lr = scheduler.optimizer.param_groups[0]['lr']

        if (epoch > 0 and epoch % model_configs['val_interval'] == 0):
            if validation_loader is not None:
                # Validation step
                model.eval()
                with torch.no_grad():
                    for val_batch_idx, (img_paths, imgs, widths, heights, depths, objects, infos) in tqdm(enumerate(validation_loader), total=len(validation_loader), desc=f"Validation - Epoch {epoch}/{model_configs['epochs']}"):

                        # Get the labels from the objects
                        labels = [obj[0] for objs in objects for obj in objs]
                        labels = torch.stack(labels, dim=0)
                        labels = labels.cuda() # [batch_size, 1]

                        # Move the images to the device
                        if type(imgs) == list or type(imgs) == tuple:
                            imgs = [im.cuda() for im in imgs]
                        else:
                            imgs = imgs.cuda()

                        # Forward pass on the Network
                        outputs = model(imgs)
                        pred_ids = torch.argmax(outputs, dim=1)  # Get predicted class IDs

                        # Loss calculation
                        loss = loss_fn(outputs, labels)
                        val_accuracy = (pred_ids == labels).float().mean()

                        val_running_loss.append(loss.detach().cpu().item())
                        val_acc_list.append(val_accuracy.detach().cpu().item())

                # Early stopping call
                early_stopping(np.mean(val_running_loss), model)
                if early_stopping.early_stop:
                    print("Early stopping triggered. Saving to WandB (if enabled) and exiting.")
                    # Here we should log to wandb and exit the code
                    if use_wandb:
                        log_dict.update({
                            'val_loss': np.mean(val_running_loss),
                            'val_accuracy': np.mean(val_acc_list),
                        })
                        wandb.log(log_dict)
                        wandb_run.finish()
                    return

                # Log the validation results
                if use_wandb:
                    log_dict.update({
                        'val_loss': np.mean(val_running_loss),
                        'val_accuracy': np.mean(val_acc_list),
                    })
                
                print(f"Validation | Epoch {epoch}/{model_configs['epochs']} | "
                        f"Loss: {np.mean(val_running_loss):.4f} | "
                        f"Accuracy: {np.mean(val_acc_list):.4f}")

        print(f"Train | Epoch {epoch}/{model_configs['epochs']} | LR: {opt_lr:.2e}\n"
                f"\t - Train Loss: {np.mean(running_loss):.4f}\n"
                f"\t - Train Accuracy: {np.mean(acc_list):.4f}\n")

        # Log the results to WandB
        if use_wandb:
            wandb.log(log_dict)

        # Save the model every epoch
        save_model({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': np.mean(running_loss),
        }, model_configs['output_dir'], model_path=f'model_ep-{epoch}_loss-{np.mean(running_loss):.4f}.pth')
    
    # Finish the run and upload any remaining data.
    wandb_run.finish()

if __name__ == '__main__':
    # If a single GPU is exposed (--device nvidia.com/gpu=4), we should leave this as 0
    # If multiple GPUs are exposed, then we should set CUDA_DEVICE properly [0, 1] / [1, 2, 3]
    CUDA_DEVICE = 0
    
    # Show some infos about the GPUs
    available_gpus = torch.cuda.device_count()
    print("Available GPUs:", available_gpus)
    for i in range(available_gpus):
        print(f"\t- GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Set-up CUDA based on how many GPUs are there
    if type(CUDA_DEVICE) == list:
        devices = ",".join(map(str, CUDA_DEVICE))
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        print(f"Setting CUDA devices with IDs: {devices}..")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
        torch.cuda.set_device(CUDA_DEVICE) # YOLO Single CUDA device selector (maybe yolo works this way and not with the CUDA_VISIBLE_DEVICES variable above)
        print(f"Setting CUDA devices with ID: {str(CUDA_DEVICE)}..")
    
    print(f"Successfully set CUDA device with ID: {torch.cuda.current_device()}")
    
    main()
