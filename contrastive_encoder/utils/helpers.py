"""
Utility Helpers
===============
Common utility functions for seeding, device management, and I/O.
"""

import os
import json
import random
import hashlib
from datetime import datetime
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def set_seed(seed: int = 42) -> Dict[str, int]:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        
    Returns:
        Dictionary with seed values for logging
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    return {
        "python_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "cuda_seed": seed if torch.cuda.is_available() else None,
    }


def get_device(prefer_gpu: bool = True, gpu_id: int = 0) -> torch.device:
    """
    Get the appropriate compute device.
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        gpu_id: Which GPU to use if multiple available
        
    Returns:
        torch.device object
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        indent: JSON indentation level
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    
    # Convert numpy arrays and other non-serializable types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return obj
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, default=convert)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_file_hash(filepath: str, algorithm: str = "md5") -> str:
    """
    Compute hash of a file for versioning.
    
    Args:
        filepath: Path to file
        algorithm: Hash algorithm ('md5' or 'sha256')
        
    Returns:
        Hex digest string (first 8 chars)
    """
    if not os.path.exists(filepath):
        return "none"
    
    hasher = hashlib.md5() if algorithm == "md5" else hashlib.sha256()
    
    with open(filepath, 'rb') as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest()[:8]


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def ensure_dir(path: str) -> str:
    """Ensure directory exists, create if not."""
    os.makedirs(path, exist_ok=True)
    return path


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(n: int) -> str:
    """Format large numbers with K/M suffixes."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
