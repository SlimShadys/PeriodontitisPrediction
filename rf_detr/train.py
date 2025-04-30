import os
import sys
from typing import Dict, Optional

import torch
import wandb
import wandb.wandb_run

from rfdetr import RFDETRBase

# Local imports
sys.path.append("./")
from misc.utils import run_checks, get_dataset, get_model_path

def train_model(model_configs: Dict, dataset: Dict, wandb_run: Optional[wandb.wandb_run.Run]) -> None:
    """
    Trains an RF-DETR model using the specified configurations and dataset.

    Args:
        model_configs (Dict): Configuration for the RF-DETR model.
        dataset (Dict): Dataset configurations.
        wandb_run (Optional[wandb.wandb_run.Run]): Active WandB run for logging.
    """
    # Extract model configs
    device = model_configs["device"]
    imgsz = model_configs["imgsz"]
    epochs = model_configs["epochs"]
    batch_size = model_configs["batch_size"]
    grad_accum_steps = model_configs["grad_accum_steps"]
    lr = model_configs["lr"]
    lr_encoder = model_configs["lr_encoder"]
    weight_decay = model_configs["weight_decay"]
    use_ema = model_configs["use_ema"]
    gradient_checkpointing = model_configs["gradient_checkpointing"]
    checkpoint_interval = model_configs["checkpoint_interval"]
    early_stopping = model_configs["early_stopping"]
    early_stopping_patience = model_configs["early_stopping_patience"]
    early_stopping_min_delta = model_configs["early_stopping_min_delta"]
    early_stopping_use_ema = model_configs["early_stopping_use_ema"]
    resume = model_configs["resume"]
    save_dir = model_configs["save_dir"]

    # Get dataset and data path
    dataset_name = dataset["name"]
    data_path = dataset["path"]
    enhance = dataset["enhance_images"]

    # Update the WandB configs before training
    if wandb_run is not None:
        wandb_run.config.update({
            "dataset_name": dataset_name,
            "device": device,
            "imgsz": imgsz,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "lr": lr,
            "lr_encoder": lr_encoder,
            "weight_decay": weight_decay,
            "use_ema": use_ema,
            "gradient_checkpointing": gradient_checkpointing,
            "checkpoint_interval": checkpoint_interval,
            "early_stopping": early_stopping,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "early_stopping_use_ema": early_stopping_use_ema,
            "enhance": enhance,
            "resume": resume,
            "save_dir": save_dir
        })

    # Initialize the RF-DETR model
    model = RFDETRBase()

    # Define training parameters
    train_params = {
        "dataset_dir": os.path.join(data_path, "RFDETR_dataset"),
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "lr": lr,
        "lr_encoder": lr_encoder,
        "resolution": imgsz,
        "weight_decay": weight_decay,
        "device": device,
        "use_ema": use_ema,
        "gradient_checkpointing": gradient_checkpointing,
        "checkpoint_interval": checkpoint_interval,
        "early_stopping": early_stopping,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "early_stopping_use_ema": early_stopping_use_ema,
        "tensorboard": True,
        "wandb": wandb_run is not None,
        "project": wandb_run.project if wandb_run else None,
        "run": wandb_run.name if wandb_run else None,
        "output_dir": save_dir
    }

    # Include resume checkpoint if specified
    if resume:
        train_params["resume"] = resume

    # Train the model
    model.train(**train_params)

def main():
    # ============== PARAMETERS ============== #

    # WandB configs
    use_wandb = True
    if use_wandb:
        wandb.login()
        entity = "SlimShadys"
        project = "FIS2-YOLODetection"
        group = "v2.2"
        wandb_id = "d3fafg8d"
    else:
        version = "v2.2"

    dataset_configs = {
        'name': 'Periapical',
        'task_type': 'detection',
        'path': os.path.join(os.getcwd(), "data", "Periapical Dataset", "Periapical Lesions"),
        'create_yolo_version': False,
        'create_rf_detr_version': False,
        'enhance_images': False,
    }

    model_configs = {
        'device': ','.join(map(str, CUDA_DEVICE)) if isinstance(CUDA_DEVICE, list) else f'cuda:{CUDA_DEVICE}',
        'imgsz': 1344,
        'epochs': 50,
        'batch_size': 4,
        'grad_accum_steps': 4,
        'lr': 1e-4,
        'lr_encoder': 1e-5,
        'weight_decay': 5e-4,
        'use_ema': True,
        'gradient_checkpointing': True,
        'checkpoint_interval': 5,
        'early_stopping': True,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.01,
        'early_stopping_use_ema': True,
        'resume': False,
    }

    # Run checks
    run_checks(model_version="RFDETR", size_version="base", dataset_configs=dataset_configs)

    # Get the dataset
    dataset_configs['data'] = get_dataset(dataset_configs)

    # Init WandB
    if use_wandb:
        if model_configs["resume"] and wandb_id is not None:
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

    # Update the WandB configs
    if use_wandb:
        model_configs["save_dir"] = os.path.join(os.getcwd(), "runs", "detect", group, wandb_run.name)
    else:
        model_configs["save_dir"] = os.path.join(os.getcwd(), "runs", "detect", version)

    # Train the model
    train_model(model_configs, dataset_configs, wandb_run)

if __name__ == '__main__':
    # If a single GPU is exposed (--device nvidia.com/gpu=4), we should leave this as 0
    # If multiple GPUs are exposed, then we should set CUDA_DEVICE properly [0, 1] / [1, 2, 3]
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
        # YOLO Single CUDA device selector (maybe yolo works this way and not with the CUDA_VISIBLE_DEVICES variable above)
        torch.cuda.set_device(CUDA_DEVICE)
    
        print(f"Successfully set CUDA device with ID: {torch.cuda.current_device()}")

    main()
