import os
import sys
from pathlib import Path

sys.path.append("./") # Needed for local imports

def run_checks(model_version, size_version, task_type="", dataset_configs=None):
    """
    Validate YOLO model configuration based on version, model size, and task type.
    
    Args:
        model_version: int or str - The YOLO version (3, 5, 8, 9, 10, 11, 12, etc.)
        size_version: str - Model size/variant (n, s, m, l, x, t, c, e, b)
        task_type: str - Task suffix like "", "-cls", "-seg", etc.
        dataset_configs: dict - Dataset configurations
    
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If the configuration is not supported
    """
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
    else:
        raise ValueError(f"Unsupported YOLO version: {model_version}")
    
    # Check if model version is valid for this family
    if size_version not in valid_configs[model_family]["sizes"]:
        raise ValueError(f"Unsupported model size '{size_version}' for {model_family}")
        
    # Check if the task type is supported for this model family
    if task_type and task_type not in valid_configs[model_family]["suffixes"]:
        raise ValueError(f"Unsupported task type '{task_type}' for {model_family}")  

    # All checks passed
    return True

def get_dataset(dataset_configs):
    dataset_name = dataset_configs['name']
    dataset_path = dataset_configs['path']
    task_type = dataset_configs.get('task_type', '')

    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    if task_type == "segmentation":
        if dataset_name == "TeethSeg":
            # =================================================================================================
            # Teeth Seg Segmentation Dataset
            # =================================================================================================
            import datasets.teeth_seg as teeth_seg
            return teeth_seg.TeethSeg(dataset_configs=dataset_configs)
        else:
            raise ValueError(f"Unsupported dataset for segmentation: {dataset_name}")

    elif task_type == "detection":
        if dataset_name == "DENTEX":
            # =================================================================================================
            # Dentex Dataset
            # =================================================================================================
            import datasets.dentex as dentex
            return dentex.Dentex(dataset_configs=dataset_configs)
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
