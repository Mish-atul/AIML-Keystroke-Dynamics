"""
PyTorch Datasets for Keystroke Dynamics
========================================
Dataset classes for supervised and contrastive training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable

from .augmentations import Compose, TwoViewAugmentation, get_contrastive_augmentations


class KeystrokeDataset(Dataset):
    """
    Standard PyTorch Dataset for supervised keystroke training.
    
    Returns (sequence, label) pairs.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            X: Sequence data of shape (N, 11, 3)
            y: Labels of shape (N,)
            transform: Optional transform to apply to sequences
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform is not None:
            # Convert to numpy, apply transform, convert back
            x_np = x.numpy()
            x_np = self.transform(x_np)
            x = torch.FloatTensor(x_np)
        
        return x, y


class ContrastiveKeystrokeDataset(Dataset):
    """
    Dataset for contrastive learning.
    
    Returns two augmented views of each sample for NT-Xent or SupCon loss.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        augmentation: Optional[Compose] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            X: Sequence data of shape (N, 11, 3)
            y: Labels of shape (N,) - optional, needed for SupCon
            augmentation: Augmentation pipeline (uses defaults if None)
            seed: Random seed for augmentations
        """
        self.X = X.astype(np.float32)
        self.y = y
        
        if augmentation is None:
            augmentation = get_contrastive_augmentations(seed=seed)
        
        self.two_view = TwoViewAugmentation(augmentation, seed=seed)
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.X[idx]
        
        # Create two augmented views
        view1, view2 = self.two_view(x)
        
        result = {
            "view1": torch.FloatTensor(view1),
            "view2": torch.FloatTensor(view2),
            "idx": torch.LongTensor([idx]),
        }
        
        if self.y is not None:
            result["label"] = torch.LongTensor([self.y[idx]])
        
        return result


class FewShotDataset(Dataset):
    """
    Dataset for few-shot adapter training.
    
    Provides support (enrollment) and query samples for a specific user.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_user: int,
        n_support: int = 5,
        n_query: int = 10,
        n_negatives: int = 32,
        seed: Optional[int] = None,
    ):
        """
        Args:
            X: Sequence data of shape (N, 11, 3)
            y: Labels of shape (N,)
            target_user: Target user label for few-shot
            n_support: Number of support (enrollment) samples
            n_query: Number of query samples for this user
            n_negatives: Number of negative samples from other users
            seed: Random seed for sampling
        """
        self.X = X.astype(np.float32)
        self.y = y
        self.target_user = target_user
        self.n_support = n_support
        self.n_query = n_query
        self.n_negatives = n_negatives
        self.rng = np.random.RandomState(seed)
        
        # Split target user samples into support and query
        target_mask = y == target_user
        target_indices = np.where(target_mask)[0]
        
        if len(target_indices) < n_support + n_query:
            raise ValueError(
                f"User {target_user} has only {len(target_indices)} samples, "
                f"need at least {n_support + n_query}"
            )
        
        # Randomly select support and query indices
        self.rng.shuffle(target_indices)
        self.support_indices = target_indices[:n_support]
        self.query_indices = target_indices[n_support:n_support + n_query]
        
        # Get negative indices (other users)
        negative_mask = ~target_mask
        self.negative_indices = np.where(negative_mask)[0]
        
    def get_support_set(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the support (enrollment) set."""
        X_support = torch.FloatTensor(self.X[self.support_indices])
        y_support = torch.LongTensor(self.y[self.support_indices])
        return X_support, y_support
    
    def get_query_set(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the query set (for evaluation)."""
        X_query = torch.FloatTensor(self.X[self.query_indices])
        y_query = torch.LongTensor(self.y[self.query_indices])
        return X_query, y_query
    
    def sample_negatives(self, n: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample negative examples from other users."""
        if n is None:
            n = self.n_negatives
        
        neg_idx = self.rng.choice(self.negative_indices, size=min(n, len(self.negative_indices)), replace=False)
        X_neg = torch.FloatTensor(self.X[neg_idx])
        y_neg = torch.LongTensor(self.y[neg_idx])
        return X_neg, y_neg
    
    def __len__(self) -> int:
        return len(self.query_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a query sample with sampled negatives."""
        query_idx = self.query_indices[idx]
        
        # Sample some negatives for this query
        neg_idx = self.rng.choice(
            self.negative_indices, 
            size=min(self.n_negatives, len(self.negative_indices)), 
            replace=False
        )
        
        return {
            "query": torch.FloatTensor(self.X[query_idx]),
            "query_label": torch.LongTensor([self.y[query_idx]]),
            "negatives": torch.FloatTensor(self.X[neg_idx]),
            "negative_labels": torch.LongTensor(self.y[neg_idx]),
        }


class TripletDataset(Dataset):
    """
    Dataset for triplet loss training.
    
    Returns (anchor, positive, negative) triplets.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        samples_per_class: int = 2,
        seed: Optional[int] = None,
    ):
        """
        Args:
            X: Sequence data of shape (N, 11, 3)
            y: Labels of shape (N,)
            samples_per_class: Number of samples per class in a batch
            seed: Random seed
        """
        self.X = X.astype(np.float32)
        self.y = y
        self.samples_per_class = samples_per_class
        self.rng = np.random.RandomState(seed)
        
        # Group indices by class
        self.class_indices = {}
        for label in np.unique(y):
            self.class_indices[label] = np.where(y == label)[0]
        
        self.classes = list(self.class_indices.keys())
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get anchor, positive, and negative samples."""
        anchor = self.X[idx]
        anchor_label = self.y[idx]
        
        # Get positive (same class, different sample)
        pos_indices = self.class_indices[anchor_label]
        pos_indices = pos_indices[pos_indices != idx]
        pos_idx = self.rng.choice(pos_indices)
        positive = self.X[pos_idx]
        
        # Get negative (different class)
        neg_label = self.rng.choice([c for c in self.classes if c != anchor_label])
        neg_idx = self.rng.choice(self.class_indices[neg_label])
        negative = self.X[neg_idx]
        
        return {
            "anchor": torch.FloatTensor(anchor),
            "positive": torch.FloatTensor(positive),
            "negative": torch.FloatTensor(negative),
            "anchor_label": torch.LongTensor([anchor_label]),
        }


def create_dataloaders(
    train_data: Dict[str, np.ndarray],
    val_data: Optional[Dict[str, np.ndarray]] = None,
    batch_size: int = 128,
    contrastive: bool = True,
    num_workers: int = 0,
    seed: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for training and validation.
    
    Args:
        train_data: Dictionary with 'X_seq' and 'y' keys
        val_data: Optional validation data
        batch_size: Batch size
        contrastive: Whether to use ContrastiveKeystrokeDataset
        num_workers: Number of data loading workers
        seed: Random seed
        
    Returns:
        Dictionary with 'train' and optionally 'val' DataLoaders
    """
    if contrastive:
        train_dataset = ContrastiveKeystrokeDataset(
            train_data["X_seq"],
            train_data.get("y"),
            seed=seed,
        )
    else:
        train_dataset = KeystrokeDataset(
            train_data["X_seq"],
            train_data["y"],
        )
    
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,  # Important for contrastive learning
        )
    }
    
    if val_data is not None:
        if contrastive:
            val_dataset = ContrastiveKeystrokeDataset(
                val_data["X_seq"],
                val_data.get("y"),
                seed=seed,
            )
        else:
            val_dataset = KeystrokeDataset(
                val_data["X_seq"],
                val_data["y"],
            )
        
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    
    return loaders
