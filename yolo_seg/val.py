import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
import wandb
import wandb.wandb_run
from ultralytics import YOLO

# Local imports
sys.path.append("./")
from misc.utils import run_checks, get_dataset, get_model_path, print_cool_ascii

def validate_model(model_configs: Dict, dataset: Dict, wandb_run: Optional[wandb.wandb_run.Run]) -> None:
    """
    Validates a YOLO model using the specified configurations and dataset.

    Args:
        model_configs (Dict): Configuration for the YOLO model.
        dataset (Dict): Dataset configurations.
        wandb_run (Optional[wandb.wandb_run.Run]): Active WandB run for logging.
    """
    # Extract model configs
    model_path = model_configs["model_path"]
    device = model_configs["device"]
    imgsz = model_configs["imgsz"]
    batch = model_configs["batch"]
    conf = model_configs["conf"]
    save_json = model_configs["save_json"]
    verbose = model_configs["verbose"]
    
    # Get dataset and data path
    dataset_name = dataset["name"]
    data_path = dataset["path"]
    data = os.path.join(data_path, "YOLO_dataset", "data.yaml")
    
    # Update the WandB configs before validation
    if wandb_run is not None:
        wandb_run.config.update({
            "model_path": model_path,
            "dataset_name": dataset_name,
            "device": device,
            "imgsz": imgsz,
            "batch": batch,
            "conf": conf,
            "save_json": save_json,
            "verbose": verbose
        })

    # Load the YOLO model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    val_args = {
        "data": data,
        "device": device,
        "imgsz": imgsz,
        "batch": batch,
        "conf": conf,
        "save_json": save_json,
        "verbose": verbose,
        'overlap_mask': True,  # Enable mask overlap for segmentation tasks
        'workers': 0,  # Number of workers for data loading
    }

    # Validate the model
    results = model.val(**val_args)
    
    # Print validation results
    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    
    # Box metrics (if available)
    if hasattr(results, 'box'):
        print(f"Box mAP50: {results.box.map50:.4f}")
        print(f"Box mAP50-95: {results.box.map:.4f}")
        print(f"Box Precision: {results.box.mp:.4f}")
        print(f"Box Recall: {results.box.mr:.4f}")
    
    # Mask/Segmentation metrics (if available)
    if hasattr(results, 'seg'):
        print(f"Mask mAP50: {results.seg.map50:.4f}")
        print(f"Mask mAP50-95: {results.seg.map:.4f}")
        print(f"Mask Precision: {results.seg.mp:.4f}")
        print(f"Mask Recall: {results.seg.mr:.4f}")
        print(f"Mask IoU: {results.mean_iou:.4f}")
        print(f"Mask Dice: {results.mean_dice:.4f}")
    
    # Log to WandB if available
    if wandb_run is not None:
        log_dict = {}
        
        # Log box metrics
        if hasattr(results, 'box'):
            log_dict.update({
                'val_box_map50': results.box.map50,
                'val_box_map50_95': results.box.map,
                'val_box_precision': results.box.mp,
                'val_box_recall': results.box.mr
            })
        
        # Log segmentation metrics
        if hasattr(results, 'seg'):
            log_dict.update({
                'val_seg_map50': results.seg.map50,
                'val_seg_map50_95': results.seg.map,
                'val_seg_precision': results.seg.mp,
                'val_seg_recall': results.seg.mr
            })
        
        if log_dict:
            wandb_run.log(log_dict)
    
    return results

def main():
    # ============== PARAMETERS ============== #

    # WandB configs
    use_wandb = False                   # Set to True if you want to use WandB for logging
    
    if use_wandb:
        wandb.login()
        entity = "SlimShadys"           # Set the wandb entity
        project = "FIS2-YOLOSegmentation-Val"  # Set the wandb project
        group = "v0.3-validation"       # Set the group name
        wandb_id = None                 # Set the run ID if you want to resume a run
    else:
        version = "v0.3-validation"     # Set the version of the run (saved locally ONLY)

    dataset_configs = {
        'task_type': 'segmentation',
        # ==== TeethSeg Dataset
        #'name': 'TeethSeg',
        #'path': os.path.join(os.getcwd(), "data", "TeethSeg"),
        # ==== DualLabel Dataset
        'name': 'DualLabel',
        'path': os.path.join(os.getcwd(), "data", "DualLabel"),
        # ==================== #
        'create_yolo_version': False,   # Don't recreate dataset for validation
        'enhance_images': False,
    }

    model_configs = {
        # Model path - UPDATE THIS to your trained model
        'model_path': os.path.join(os.getcwd(), "yolo_seg", "best-glad-sound-59.pt"),
        
        # Validation configs
        'device': f'cuda:{CUDA_DEVICE}' if isinstance(CUDA_DEVICE, int) else ','.join(map(str, CUDA_DEVICE)),
        'imgsz': 1280,                  # Image size for validation
        'batch': 8,                     # Batch size for validation (can be larger than training)
        'conf': 0.25,                  # Confidence threshold (lower for validation)
        'save_json': True,              # Save results in JSON format
        'verbose': True,                # Verbose output
    }

    # Get the dataset
    dataset_configs['data'] = get_dataset(dataset_configs)

    # Init WandB
    if use_wandb:
        wandb_run = wandb.init(
            entity=entity,
            project=project,
            group=group
        )
    else:
        wandb_run = None

    # Run checks
    run_checks(model_version='12-turbo', size_version='m', dataset_configs=dataset_configs)

    # Validate the model
    validate_model(model_configs, dataset_configs, wandb_run)
    
    # Finish WandB run
    if wandb_run is not None:
        wandb_run.finish()

if __name__ == '__main__':
    # GPU Configuration (same as train.py)
    CUDA_DEVICE = 0

    available_gpus = torch.cuda.device_count()
    print("Available GPUs:", available_gpus)
    for i in range(available_gpus):
        print(f"\t- GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Set-up CUDA based on how many GPUs are there
    if type(CUDA_DEVICE) == list:
        devices = ",".join(map(str, CUDA_DEVICE))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, CUDA_DEVICE))
        print(f"Setting CUDA devices with IDs: {devices}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
        torch.cuda.set_device(CUDA_DEVICE)
        print(f"Successfully set CUDA device with ID: {torch.cuda.current_device()}")
    
    main()

    # Print completion message
    print("\n\nValidation completed successfully! 🎉")
    print_cool_ascii()
