import os
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.append("./") # Needed for local imports

def run_checks(model_version, size_version, dataset_configs=None):
    """
    Validate YOLO model configuration based on version, model size, and task type.
    
    Args:
        model_version: int or str - The YOLO version (3, 5, 8, 9, 10, 11, 12, etc.)
        size_version: str - Model size/variant (n, s, m, l, x, t, c, e, b)
        dataset_configs: dict - Dataset configurations
    
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If the configuration is not supported
    """
    # Retrieve the task type from dataset_configs if provided
    if dataset_configs is not None:
        task_type = dataset_configs.get('task_type', '')
    else:
        raise ValueError("Dataset configuration file must be provided to determine task type")
    
    # Define valid configurations as a dictionary
    valid_configs = {
        "yolov8": {
            "sizes": ["n", "s", "m", "l", "x"],
            "suffixes": ["", "-cls", "-seg", "-pose", "-obb", "-oiv7", "-world", "-worldv2"]
        },
        "yolov9": {
            "sizes": ["t", "s", "m", "c", "e"],
            "suffixes": ["", "-seg"]
        },
        "yolov10": {
            "sizes": ["n", "s", "m", "b", "l", "x"],
            "suffixes": [""]
        },
        "yolo11": {
            "sizes": ["n", "s", "m", "l", "x"],
            "suffixes": ["", "-cls", "-seg", "-pose", "-obb"]
        },
        "yolo12": {
            "sizes": ["n", "s", "m", "l", "x"],
            "suffixes": [""]  # Currently Detect-models only as per https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/downloads.py#L21
        },
        "yoloe-v8": {
            "sizes": ["s", "m", "l"],
            "suffixes": ["-seg", "-seg-pf"]
        },
        "yoloe-11": {
            "sizes": ["s", "m", "l"],
            "suffixes": ["-seg", "-seg-pf"]
        },
        "rfdetr": {
            "sizes": ["base", "large"],
            "suffixes": [""]
        }
    }

    # Determine the model family
    if isinstance(model_version, int):
        model_version = str(model_version)
        
    if model_version == "8" or model_version == "9" or model_version == "10":
        model_family = f"yolov{model_version}"
    elif model_version == "11" or model_version == "12":
        model_family = f"yolo{model_version}"
    elif model_version in ["yoloe-v8", "yoloe-11"]:
        model_family = model_version
    elif model_version == "RFDETR":
        model_family = "rfdetr"
    else:
        raise ValueError(f"Unsupported YOLO version: {model_version}")
    
    # Check if model version is valid for this family
    if size_version not in valid_configs[model_family]["sizes"]:
        raise ValueError(f"Unsupported model size '{size_version}' for {model_family}")
        
    # Check if the task type is supported for this model family
    if task_type:
        if task_type == 'detection': task_type = ''
        elif task_type == 'segmentation': task_type = '-seg'
        elif task_type == 'classification': task_type = '-cls'
        elif task_type == 'pose': task_type = '-pose'
        elif task_type == 'obb': task_type = '-obb'
        elif task_type == 'oiv7': task_type = '-oiv7'
        elif task_type == 'world': task_type = '-world'
        elif task_type == 'worldv2': task_type = '-worldv2'
        elif task_type == 'segmentation-pf': task_type = '-seg-pf'
        else:
            raise ValueError(f"Unsupported task type '{task_type}'")
        
        if task_type not in valid_configs[model_family]["suffixes"]:
            raise ValueError(f"Unsupported task type '{task_type}' for {model_family}")  

    # All checks passed
    return True

def get_dataset(dataset_configs):
    dataset_name = dataset_configs['name']
    dataset_path = dataset_configs['path']
    task_type = dataset_configs.get('task_type', '')

    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Dataset path '{dataset_path}' does not exist.")

    if task_type == "segmentation":
        if dataset_name == "TeethSeg":
            # =================================================================================================
            # Teeth Seg Segmentation Dataset
            # =================================================================================================
            import datasets.teeth_seg as teeth_seg
            return teeth_seg.TeethSeg(dataset_configs=dataset_configs)
        elif dataset_name == "DualLabel":
            import datasets.dual_label as dual_label
            return dual_label.DualLabel(dataset_configs=dataset_configs)
        else:
            raise ValueError(f"Unsupported dataset for segmentation: {dataset_name}")

    elif task_type == "detection":
        if dataset_name == "DENTEX":
            # =================================================================================================
            # Dentex Dataset
            # =================================================================================================
            import datasets.dentex as dentex
            return dentex.Dentex(dataset_configs=dataset_configs)
        elif dataset_name == "Periapical":
            # =================================================================================================
            # Periapical Dataset
            # =================================================================================================
            import datasets.periapical as periapical
            return periapical.PeriapicalDatasetDet(dataset_configs=dataset_configs)
        else:
            raise ValueError(f"Unsupported dataset for detection: {dataset_name}")
    else:
        raise ValueError(f"Unsupported task: {task_type}")

def get_model_path(resume, save_dir, model_version, size_version, task_type="-seg"):
    """
    Get the appropriate model path based on resume flag and model configuration.
    
    Args:
        resume: bool - Whether to resume from an existing checkpoint
        save_dir: str - Directory where checkpoints are saved
        model_version: int or str - The YOLO version (8, 9, 10, 11, 12, etc.)
        size_version: str - Model size/variant (n, s, m, l, x, t, c, e, b)
        task_type: str - Task suffix like "", "-cls", "-seg", etc.
    
    Returns:
        str: Path to the model weights
    
    Raises:
        FileNotFoundError: If resume is True but checkpoint file doesn't exist
    """
    if resume:
        model_path = os.path.join(save_dir, "weights", "last.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint file not found: {model_path}")
    else:
        # Determine the model name format based on version
        if isinstance(model_version, int):
            model_version = str(model_version)
            
        # Construct model name based on version convention
        if model_version in ["8", "9", "10"]:
            model_name = f"yolov{model_version}{size_version}{task_type}.pt"
        elif model_version in ["11", "12"]:
            model_name = f"yolo{model_version}{size_version}{task_type}.pt"
        elif model_version == "yoloe-v8":
            model_name = f"yoloe-v8{size_version}{task_type}.pt"
        elif model_version == "yoloe-11":
            model_name = f"yoloe-11{size_version}{task_type}.pt"
        else:
            raise ValueError(f"Unsupported YOLO version: {model_version}")
            
        model_path = model_name
    
    return model_path

def is_docker() -> bool:
    cgroup = Path('/proc/self/cgroup')
    return Path('/.dockerenv').is_file() or (cgroup.is_file() and 'docker' in cgroup.read_text())

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def save_model(state, output_dir, model_path='model.pth'):
    '''
        Save the model to a file
        @param state: The state of the model, optimizer, scheduler, and loss value
        @param output_dir: The output directory to save the model
        @param model_path: The file name of the model to save
    '''
    torch.save(state, os.path.join(output_dir, model_path))
    
def load_model(model_path: str,
               model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler_warmup: torch.optim.lr_scheduler._LRScheduler,
               device):
    '''
    Load the model from a file
    @param model_path: The path to the model file
    @param model: The model to load
    @param optimizer: The optimizer to load
    @param scheduler_warmup: The scheduler to load
    @param device: The device to load the model to
    '''
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler_warmup.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Correctly loaded the model, optimizer, and scheduler from: {model_path}")
    return model, optimizer, scheduler_warmup, start_epoch
