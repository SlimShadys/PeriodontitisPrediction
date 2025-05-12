import os
import sys
from typing import Dict, Optional

import torch
import wandb
import wandb.wandb_run
from ultralytics import YOLO, YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPTrainer

# Local imports
sys.path.append("./")
from misc.utils import run_checks, get_dataset, get_model_path

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
    lrf = model_configs["lrf"]
    cos_lr = model_configs["cos_lr"]
    close_mosaic = model_configs["close_mosaic"]
    momentum = model_configs["momentum"]
    weight_decay = model_configs["weight_decay"]
    patience = model_configs["patience"]
    dropout = model_configs["dropout"]
    mask_ratio = model_configs["mask_ratio"]
    overlap_mask = model_configs["overlap_mask"]
    warmup_epochs = model_configs["warmup_epochs"]
    cache = model_configs["cache"]
    
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
    enhance = dataset["enhance_images"]
    #data = dataset["data"]

    # Get model path
    # Last parameter is the task type:
    #   - "" for detection
    #   - "-seg" for segmentation
    #   - "-cls" for classification
    model_path = get_model_path(resume, save_dir, yolo_version, model_version, '-seg')
        
    # Load the YOLO model
    if model_path.startswith("yoloe"):
        raise NotImplementedError("YOLOE for Segmentation training is not implemented yet.")
        # Load YOLOE model
        model = YOLOE(model_path)
        # freeze every layer except of the savpe module.
        head_index = len(model.model.model) - 1
        freeze = list(range(0, head_index))
        for name, child in model.model.model[-1].named_children():
            if "savpe" not in name:
                freeze.append(f"{head_index}.{name}")
        trainer = YOLOEVPTrainer
        # In this case, Yolo-E expects a dictionary and not a .yaml file
        # The reason behind this is that we need the grounding_data which is not present in the TeethSeg dataset (for now)
        data = None
        # data = dict(
        #     train=dict(
        #         yolo_data=[os.path.join(data_path, "YOLO_dataset", "data.yaml")],
        #         grounding_data=[
        #             dict(
        #                 img_path="../datasets/flickr/full_images/",
        #                 json_file="../datasets/flickr/annotations/final_flickr_separateGT_train_segm.json",
        #             ),
        #             # dict(
        #             #     img_path="../datasets/mixed_grounding/gqa/images",
        #             #     json_file="../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
        #             # ),
        #         ],
        #     ),
        #     val=dict(os.path.join(data_path, "YOLO_dataset", "data.yaml")),
        # )
    else:
        if yolo_version == "12-turbo" or yolo_version == "12":
            # Load YOLOv12-turbo model
            model = YOLO(f"yolov12{model_version}-seg.yaml").load(f"yolo12{model_version}-seg.pt") # --> Requires YOLOv12-turbo and yolov12-seg.yaml in the configs folder
        else:
            model = YOLO(model_path)
        trainer = model.trainer
        freeze = None
        data = os.path.join(data_path, "YOLO_dataset", "data.yaml")

    # Update the WandB configs before training
    if wandb_run is not None:
        wandb_run.config.update({"yolo_version": yolo_version, "model_version": model_version, "dataset_name": dataset_name,
            "device": device, "imgsz": imgsz, "epochs": epochs, "batch": batch, "optimizer": optimizer,
            "lr0": lr0, "lrf": lrf, "cos_lr": cos_lr, "close_mosaic": close_mosaic,
            "momentum": momentum, "weight_decay": weight_decay, "patience": patience, "augmentations": augmentations, "enhance": enhance,
            "mask_ratio": mask_ratio, "overlap_mask": overlap_mask, "dropout": dropout, "warmup_epochs": warmup_epochs, "cache": cache,
            "resume": resume, "save_dir": save_dir, "model_path": model_path
        })

    # Train the model
    results = model.train(data=data,
        # Trainer-related
        trainer=trainer, freeze=freeze, cache=cache, warmup_epochs=warmup_epochs,
        # Network-related
        device=device, imgsz=imgsz, epochs=epochs, batch=batch, optimizer=optimizer, lr0=lr0, lrf=lrf,
        cos_lr=cos_lr, momentum=momentum, weight_decay=weight_decay, patience=patience,
        dropout=dropout, mask_ratio=mask_ratio, overlap_mask=overlap_mask, 
        # Augmentations
        hsv_h=augmentations["hsv_h"], hsv_s=augmentations["hsv_s"], hsv_v=augmentations["hsv_v"],
        degrees=augmentations["degrees"], fliplr=augmentations["fliplr"], close_mosaic=close_mosaic,
        # WandB configs and resume option
        project=os.path.join("runs", "segment", wandb_run.group) if wandb_run is not None else os.sep.join(save_dir.split(os.sep)[:-1]),
        name=wandb_run.name if wandb_run is not None else save_dir.split(os.sep)[-1],
        resume=resume,
    )

def main():
    # ============== PARAMETERS ============== #

    # WandB configs
    use_wandb = True                    # Set to True if you want to use WandB for logging | # If True, also run ```yolo settings wandb=True``` in the terminal
    
    if use_wandb:
        wandb.login()
        entity = "SlimShadys"           # Set the wandb entity where your project will be logged (generally your team name).
        project = "FIS2-YOLOSegmentation"  # Set the wandb project where this run will be logged.
        group = "v0.3"                  # Set the group name of the run, this is useful when comparing multiple runs in a project.
        wandb_id = None           # Set the run ID if you want to resume a run
    else:
        version = "v0.3"                # Set the version of the run (saved locally ONLY)

    dataset_configs = {
        'task_type': 'segmentation',
        # ==== TeethSeg Dataset
        #'name': 'TeethSeg',
        #'path': os.path.join(os.getcwd(), "data", "TeethSeg"), # For local testing
        #'path': os.path.abspath(os.path.join(os.getcwd(), "..", "datasets", "TeethSeg")), # For Docker testing
        # ==== DualLabel Dataset
        'name': 'DualLabel',
        'path': os.path.join(os.getcwd(), "data", "DualLabel"), # For local testing
        #'path': os.path.abspath(os.path.join(os.getcwd(), "..", "datasets", "DualLabel")), # For Docker testing
        # ==================== #
        'create_yolo_version': True, # Create the YOLO version of the dataset
        'enhance_images': False, # Apply image enhancements (sharpening, contrast, gaussian filtering) - Useless if create_yolo_version is False
    }

    model_configs = {
        'yolo_version': "8",      # Choose between [8, 9, 10, 11, 12, 12-turbo]. SPECIAL VERSIONS: [yoloe-v8, yoloe-11]
        'model_version': 'm',   # Choose between [n, s, m, l, x, t, c, e, b]
        'device': ','.join(map(str, CUDA_DEVICE)) if isinstance(CUDA_DEVICE, list) else f'cuda:{CUDA_DEVICE}',
        'cache': True, # Cache images for faster training
        # Network-related
        'imgsz': 1280,
        'epochs': 80,
        'batch': 4,
        'optimizer': 'auto',  # Choose between [SGD, Adam, AdamW, RMSProp, auto]
        'lr0': 1e-2,
        'lrf': 1e-2,
        'cos_lr': False,
        'close_mosaic': 10,
        'momentum': 0.937,
        'weight_decay': 5e-4,
        'patience': 20,
        'dropout': 0.0, # Dropout rate (0.0 = no dropout) | Default: 0.1
        'warmup_epochs': 0, # Warmup epochs (0 = no warmup) | Default: 3
        'mask_ratio': 4,
        'overlap_mask': True,
        # Augmentations
        'hsv_h': 0.0,           # Image HSV-Hue augmentation (fraction)
        'hsv_s': 0.0,           # Image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.1,           # Image HSV-Value augmentation (fraction)
        'degrees': 0.0,         # Image rotation (+/- deg)
        'fliplr': 0.0,          # Horizontal flip probability
        # Resume option
        'resume': False,        # Resume training from the last checkpoint (last.pt)
    }

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
        model_configs["save_dir"] = os.path.join(os.getcwd(), "runs", "segment", group, wandb_run.name) # Save directory
    else:
        model_configs["save_dir"] = os.path.join(os.getcwd(), "runs", "segment", version) # Save directory

    # Run checks
    run_checks(model_version=model_configs["yolo_version"], size_version=model_configs["model_version"], dataset_configs=dataset_configs)

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
