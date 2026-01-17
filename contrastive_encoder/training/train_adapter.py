"""
Adapter Training Module
=======================
Few-shot adapter training for per-user personalization.

Given:
- A frozen encoder
- k enrollment samples from a user

Trains a small adapter MLP and computes the user's centroid.
"""

import os
from typing import Dict, Optional, Any, List
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from ..models.encoder import KeystrokeEncoder
from ..models.adapter import UserAdapter, UserTemplate, create_adapter, compute_centroid
from ..models.losses import TripletLoss
from ..utils.helpers import (
    set_seed,
    get_device,
    save_json,
    compute_file_hash,
    AverageMeter,
    EarlyStopping,
)
from ..config import AdapterConfig


class AdapterTrainer:
    """
    Trainer for per-user adapters.
    """
    
    def __init__(
        self,
        encoder: KeystrokeEncoder,
        config: Optional[AdapterConfig] = None,
        device: Optional[torch.device] = None,
        encoder_path: Optional[str] = None,
    ):
        """
        Args:
            encoder: Pretrained encoder (will be frozen)
            config: Adapter configuration
            device: Compute device
            encoder_path: Path to encoder file (for hash tracking)
        """
        self.config = config if config is not None else AdapterConfig()
        self.device = device if device is not None else get_device()
        
        # Load and freeze encoder
        self.encoder = encoder.to(self.device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Compute encoder hash for versioning
        self.encoder_hash = "none"
        if encoder_path and os.path.exists(encoder_path):
            self.encoder_hash = compute_file_hash(encoder_path)
    
    def _compute_embeddings(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """Compute embeddings using frozen encoder."""
        with torch.no_grad():
            X = X.to(self.device)
            embeddings = self.encoder(X, normalize=True)
        return embeddings
    
    def train_adapter(
        self,
        user_id: str,
        X_support: np.ndarray,
        X_negatives: np.ndarray,
        y_support: Optional[np.ndarray] = None,
        y_negatives: Optional[np.ndarray] = None,
        X_validation: Optional[np.ndarray] = None,
    ) -> UserTemplate:
        """
        Train adapter for a single user.
        
        Args:
            user_id: User identifier
            X_support: Enrollment samples (k, 11, 3)
            X_negatives: Negative samples from other users (n, 11, 3)
            y_support: Labels for support samples (optional)
            y_negatives: Labels for negative samples (optional)
            X_validation: Validation samples for threshold calibration
            
        Returns:
            Trained UserTemplate with adapter, centroid, and threshold
        """
        n_shots = len(X_support)
        print(f"\nTraining adapter for user: {user_id}")
        print(f"Enrollment shots: {n_shots}")
        print(f"Negative samples: {len(X_negatives)}")
        
        # Convert to tensors
        X_support_t = torch.FloatTensor(X_support)
        X_negatives_t = torch.FloatTensor(X_negatives)
        
        # Compute embeddings
        support_embeddings = self._compute_embeddings(X_support_t)
        negative_embeddings = self._compute_embeddings(X_negatives_t)
        
        # Build adapter
        adapter = create_adapter(
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        ).to(self.device)
        
        print(f"Adapter parameters: {adapter.count_parameters():,}")
        
        # Build optimizer
        optimizer = optim.Adam(
            adapter.parameters(),
            lr=self.config.learning_rate,
        )
        
        # Build loss
        if self.config.loss_type == "triplet":
            loss_fn = TripletLoss(margin=self.config.triplet_margin)
        else:
            loss_fn = nn.CosineEmbeddingLoss()
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config.early_stop_patience,
            mode="min",
        )
        
        # Training loop
        adapter.train()
        best_loss = float('inf')
        best_state = None
        
        for step in range(self.config.max_steps):
            optimizer.zero_grad()
            
            # Apply adapter to support embeddings
            adapted_support = adapter(support_embeddings)
            
            # Compute centroid of adapted support
            centroid = adapted_support.mean(dim=0, keepdim=True)
            centroid = F.normalize(centroid, p=2, dim=1)
            
            # Sample hard negatives
            if self.config.use_hard_negatives:
                neg_adapted = adapter(negative_embeddings)
                # Find closest negatives to centroid
                neg_sim = F.cosine_similarity(neg_adapted, centroid)
                hard_indices = neg_sim.argsort(descending=True)[:self.config.num_negatives]
                selected_neg = negative_embeddings[hard_indices]
            else:
                # Random sample
                perm = torch.randperm(len(negative_embeddings))[:self.config.num_negatives]
                selected_neg = negative_embeddings[perm]
            
            adapted_neg = adapter(selected_neg)
            
            if self.config.loss_type == "triplet":
                # Create triplets: anchor=centroid, positive=support, negative=neg
                # Expand centroid to match support size
                anchor = centroid.expand(n_shots, -1)
                positive = adapted_support
                
                # Sample negatives for each positive
                neg_indices = torch.randint(0, len(adapted_neg), (n_shots,))
                negative = adapted_neg[neg_indices]
                
                loss = loss_fn(anchor, positive, negative)
            else:
                # Cosine embedding loss
                # Pull support towards centroid
                pos_targets = torch.ones(n_shots, device=self.device)
                loss_pos = loss_fn(adapted_support, centroid.expand(n_shots, -1), pos_targets)
                
                # Push negatives away from centroid
                neg_targets = -torch.ones(len(adapted_neg), device=self.device)
                loss_neg = loss_fn(adapted_neg, centroid.expand(len(adapted_neg), -1), neg_targets)
                
                loss = loss_pos + loss_neg
            
            loss.backward()
            optimizer.step()
            
            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}
            
            # Early stopping
            if early_stopping(loss.item()):
                print(f"  Early stopping at step {step}")
                break
            
            if step % 50 == 0:
                print(f"  Step {step}: loss = {loss.item():.4f}")
        
        # Load best state
        if best_state is not None:
            adapter.load_state_dict(best_state)
        
        adapter.eval()
        
        # Compute final centroid
        with torch.no_grad():
            adapted_support = adapter(support_embeddings)
            centroid = adapted_support.mean(dim=0)
            centroid = F.normalize(centroid, p=2, dim=0)
        
        # Compute threshold
        threshold = self._compute_threshold(
            adapter,
            support_embeddings,
            negative_embeddings,
            centroid,
            X_validation,
        )
        
        # Create template
        template = UserTemplate(
            user_id=user_id,
            adapter=adapter.cpu(),
            centroid=centroid.cpu(),
            threshold=threshold,
            enrollment_shots=n_shots,
            encoder_hash=self.encoder_hash,
            adapter_version="1.0.0",
        )
        
        print(f"  Final threshold: {threshold:.4f}")
        print(f"  Training complete")
        
        return template
    
    def _compute_threshold(
        self,
        adapter: UserAdapter,
        support_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
        centroid: torch.Tensor,
        X_validation: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute verification threshold.
        
        Uses EER (Equal Error Rate) by default.
        """
        adapter.eval()
        
        with torch.no_grad():
            # Compute similarities for support (genuine)
            adapted_support = adapter(support_embeddings)
            genuine_sims = F.cosine_similarity(
                adapted_support, 
                centroid.unsqueeze(0).expand(len(adapted_support), -1)
            ).cpu().numpy()
            
            # Compute similarities for negatives (impostor)
            adapted_neg = adapter(negative_embeddings)
            impostor_sims = F.cosine_similarity(
                adapted_neg,
                centroid.unsqueeze(0).expand(len(adapted_neg), -1)
            ).cpu().numpy()
            
            # Use validation data if provided
            if X_validation is not None:
                X_val_t = torch.FloatTensor(X_validation).to(self.device)
                val_embeddings = self._compute_embeddings(X_val_t)
                adapted_val = adapter(val_embeddings)
                val_sims = F.cosine_similarity(
                    adapted_val,
                    centroid.unsqueeze(0).expand(len(adapted_val), -1)
                ).cpu().numpy()
                genuine_sims = np.concatenate([genuine_sims, val_sims])
        
        # Find EER threshold
        thresholds = np.linspace(0, 1, 100)
        best_eer = 1.0
        best_threshold = 0.5
        
        for thresh in thresholds:
            # FAR: impostors accepted
            far = (impostor_sims >= thresh).mean()
            # FRR: genuine rejected
            frr = (genuine_sims < thresh).mean()
            
            eer_approx = abs(far - frr)
            if eer_approx < best_eer:
                best_eer = eer_approx
                best_threshold = thresh
                if eer_approx < 0.01:  # Close enough to EER
                    break
        
        return float(best_threshold)


def train_adapter_for_user(
    encoder_path: str,
    user_id: str,
    X_support: np.ndarray,
    X_negatives: np.ndarray,
    output_dir: str = "artifacts",
    config: Optional[AdapterConfig] = None,
    device: Optional[torch.device] = None,
) -> UserTemplate:
    """
    Convenience function to train adapter for a single user.
    
    Args:
        encoder_path: Path to pretrained encoder
        user_id: User identifier
        X_support: Enrollment samples (k, 11, 3)
        X_negatives: Negative samples (n, 11, 3)
        output_dir: Base output directory
        config: Adapter configuration
        device: Compute device
        
    Returns:
        Trained UserTemplate
    """
    device = device if device is not None else get_device()
    
    # Load encoder
    encoder = KeystrokeEncoder()
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    
    # Train adapter
    trainer = AdapterTrainer(
        encoder=encoder,
        config=config,
        device=device,
        encoder_path=encoder_path,
    )
    
    template = trainer.train_adapter(
        user_id=user_id,
        X_support=X_support,
        X_negatives=X_negatives,
    )
    
    # Save
    templates_dir = os.path.join(output_dir, "templates")
    adapters_dir = os.path.join(output_dir, "adapters")
    template.save(templates_dir, adapters_dir)
    
    return template


def train_adapters_for_all_users(
    encoder_path: str,
    data: Dict[str, np.ndarray],
    n_shots: int = 5,
    output_dir: str = "artifacts",
    config: Optional[AdapterConfig] = None,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Dict[str, UserTemplate]:
    """
    Train adapters for all users in the dataset.
    
    Args:
        encoder_path: Path to pretrained encoder
        data: Data dictionary with 'X_seq', 'y', 'user_ids'
        n_shots: Number of enrollment shots per user
        output_dir: Base output directory
        config: Adapter configuration
        device: Compute device
        seed: Random seed for enrollment sampling
        
    Returns:
        Dictionary mapping user_id to UserTemplate
    """
    np.random.seed(seed)
    device = device if device is not None else get_device()
    
    # Load encoder
    encoder = KeystrokeEncoder()
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    
    trainer = AdapterTrainer(
        encoder=encoder,
        config=config,
        device=device,
        encoder_path=encoder_path,
    )
    
    X_seq = data["X_seq"]
    y = data["y"]
    
    unique_users = np.unique(y)
    templates = {}
    
    print(f"\nTraining adapters for {len(unique_users)} users")
    print(f"Enrollment shots: {n_shots}")
    
    for user_label in tqdm(unique_users, desc="Users"):
        # Get user's samples
        user_mask = y == user_label
        user_indices = np.where(user_mask)[0]
        user_id = f"s{user_label:03d}"
        
        if len(user_indices) < n_shots:
            print(f"  Skipping user {user_id}: only {len(user_indices)} samples")
            continue
        
        # Sample enrollment and leave rest for validation
        np.random.shuffle(user_indices)
        support_indices = user_indices[:n_shots]
        
        X_support = X_seq[support_indices]
        
        # Negatives: samples from other users
        neg_mask = ~user_mask
        neg_indices = np.where(neg_mask)[0]
        np.random.shuffle(neg_indices)
        
        n_negatives = min(len(neg_indices), 500)  # Cap at 500
        X_negatives = X_seq[neg_indices[:n_negatives]]
        
        # Train adapter
        template = trainer.train_adapter(
            user_id=user_id,
            X_support=X_support,
            X_negatives=X_negatives,
        )
        
        # Save
        templates_dir = os.path.join(output_dir, "templates")
        adapters_dir = os.path.join(output_dir, "adapters")
        template.save(templates_dir, adapters_dir)
        
        templates[user_id] = template
    
    print(f"\nTrained {len(templates)} adapters")
    return templates
