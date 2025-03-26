import os
import sys
from typing import Dict, Optional

import wandb
import wandb.wandb_run
from ultralytics import YOLO

# Local imports
sys.path.append("./")
import datasets.dentex as dentex

def run_checks(yolo_version, model_version, dataset_configs):
    # Check if the YOLO version is supported
    if yolo_version not in [8, 9, 10, 11, 12]:
        raise ValueError(f"Unsupported YOLO version: {yolo_version}")

    # Check if the model version is supported
    if model_version not in ["n", "s", "m", "l", "x"]:
        raise ValueError(f"Unsupported model version: {model_version}")

    # Extract dataset configs
    dataset_name = dataset_configs['name']
    dataset_path = dataset_configs['path']

    # Check if the dataset is supported
    if dataset_name not in ["DENTEX"]:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

def get_dataset(dataset_configs):
    dataset_name = dataset_configs['name']

    if dataset_name == "DENTEX":
        # =================================================================================================
        # Dentex Dataset
        # - This class will convert the Dentex dataset to the YOLO format
        # - It will also create the data.yaml file needed for training
        # =================================================================================================
        return dentex.Dentex(dataset_configs=dataset_configs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_model_path(resume, save_dir, yolo_version, model_version):
    #   - If resume is True , we will load the last.pt model from the save_dir
    #   - If resume is False, we will use the default YOLO model path
    if resume:
        model_path = os.path.join(save_dir, "weights", "last.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint file not found: {model_path}")
    else:
        model_path = f"yolov{yolo_version}{model_version}.pt"
    return model_path

def train_model(model_configs: Dict, dataset: Dict, wandb_run: Optional[wandb.wandb_run.Run]) -> None:
    """
    Trains a YOLO model using the specified configurations and dataset.

    Args:
        model_configs (Dict): Configuration for the YOLO model.
        dataset (Dict): Dataset configurations.
        wandb_run (Optional[wandb.wandb_run.Run]): Active WandB run for logging.
    """
    # Extract model configs
    yolo_version = model_configs["yolo_version"]
    model_version = model_configs["model_version"]
    device = model_configs["device"]

    # Network-related
    imgsz = model_configs["imgsz"]
    epochs = model_configs["epochs"]
    batch = model_configs["batch"]
    optimizer = model_configs["optimizer"]
    lr0 = model_configs["lr0"]
    momentum = model_configs["momentum"]
    weight_decay = model_configs["weight_decay"]
    patience = model_configs["patience"]

    # Augmentations
    augmentations = {
        "hsv_h": model_configs["hsv_h"],
        "hsv_s": model_configs["hsv_s"],
        "hsv_v": model_configs["hsv_v"],
        "degrees": model_configs["degrees"],
        "fliplr": model_configs["fliplr"]
    }

    # Dirs
    resume = model_configs["resume"]
    save_dir = model_configs["save_dir"]

    # Get dataset and data path
    dataset_name = dataset["name"]
    data_path = dataset["path"]
    #data = dataset["data"]

    # Get model path
    model_path = get_model_path(resume, save_dir, yolo_version, model_version)
        
    # Load the YOLO model
    model = YOLO(model_path)

    # Update the WandB configs before training
    if wandb_run is not None:
        wandb_run.config.update({"yolo_version": yolo_version, "model_version": model_version, "dataset_name": dataset_name,
            "device": device, "imgsz": imgsz, "epochs": epochs, "batch": batch, "optimizer": optimizer,
            "lr0": lr0, "momentum": momentum, "weight_decay": weight_decay, "patience": patience, "augmentations": augmentations,
            "resume": resume, "save_dir": save_dir, "model_path": model_path
        })

    # Train the model
    results = model.train(data=os.path.join(data_path, "YOLO_dataset", "data.yaml"),
        # Network-related
        device=device, imgsz=imgsz, epochs=epochs, batch=batch, optimizer=optimizer, lr0=lr0,
        momentum=momentum, weight_decay=weight_decay, patience=patience,
        # Augmentations
        hsv_h=augmentations["hsv_h"], hsv_s=augmentations["hsv_s"], hsv_v=augmentations["hsv_v"],
        degrees=augmentations["degrees"], fliplr=augmentations["fliplr"],
        # Dirs
        resume=resume,
        # WandB
        project=os.path.join("runs", "detect", wandb_run.group) if wandb_run is not None else save_dir.split(os.sep)[:-1],
        name=wandb_run.name if wandb_run is not None else save_dir.split(os.sep)[-1],
    )

def main():
    # ============== PARAMETERS ============== #

    # WandB configs
    use_wandb = True                    # Set to True if you want to use WandB for logging | # If True, also run ```yolo settings wandb=True``` in the terminal
    if use_wandb:
        wandb.login()
        entity = "SlimShadys"           # Set the wandb entity where your project will be logged (generally your team name).
        project = "FIS2-YOLODetection"  # Set the wandb project where this run will be logged.
        group = "v2.1"                  # Set the group name of the run, this is useful when comparing multiple runs in a project.
        wandb_id = "d3fafg8d"           # Set the run ID if you want to resume a run
    else:
        version = "v2.1"                # Set the version of the run (saved locally ONLY)

    # Retrieve the last run ID for the project
    # api = wandb.Api()
    # runs = api.runs(f"{entity}/{project}")
    # last_run_id = runs[0].id  # Get the most recent run ID

    dataset_configs = {
        'name': 'DENTEX',
        'path': os.path.join(os.getcwd(), "data", "DENTEX", "DENTEX"),
        'create_yolo_version': False
    }

    model_configs = {
        'yolo_version': 8,      # Choose between [8, 9, 10, 11, 12]
        'model_version': 'm',   # Choose between [n, s, m, l, x]
        'device': 'cuda',
        # Network-related
        'imgsz': 1280,
        'epochs': 30,
        'batch': 4,
        'optimizer': 'adamw',
        'lr0': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'patience': 20,
        # Augmentations
        'hsv_h': 0.0,           # Image HSV-Hue augmentation (fraction)
        'hsv_s': 0.0,           # Image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.1,           # Image HSV-Value augmentation (fraction)
        'degrees': 0.0,         # Image rotation (+/- deg)
        'fliplr': 0.0,          # Horizontal flip probability
        # Dirs
        'resume': True,       # Resume training from the last checkpoint (last.pt)
        'save_dir': None,      # Save directory
    }

    # Run checks
    run_checks(model_configs["yolo_version"], model_configs["model_version"], dataset_configs)

    # Get the dataset
    dataset_configs['data'] = get_dataset(dataset_configs)

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

    # Update the WandB configs
    if use_wandb:
        model_configs["save_dir"] = os.path.join(os.getcwd(), "runs", "detect", group, wandb_run.name) # Save directory
    else:
        model_configs["save_dir"] = os.path.join(os.getcwd(), "runs", "detect", version) # Save directory
        
    # Train the model
    train_model(model_configs, dataset_configs, wandb_run)

if __name__ == '__main__':
    main()
