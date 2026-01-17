"""
Contrastive and Triplet Losses
==============================
Loss functions for self-supervised and supervised contrastive learning.

Implements:
- NT-Xent (Normalized Temperature-scaled Cross Entropy) - SimCLR style
- SupCon (Supervised Contrastive Loss)
- Triplet Loss with hard negative mining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class NTXentLoss(nn.Module):
    """
    NT-Xent Loss (Normalized Temperature-scaled Cross Entropy).
    
    Used in SimCLR for self-supervised contrastive learning.
    Given a batch of N samples, creates 2N augmented views.
    For each sample, treats the other augmented view as positive
    and all other samples as negatives.
    
    Reference: Chen et al., "A Simple Framework for Contrastive Learning
               of Visual Representations", ICML 2020
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter (τ) for scaling similarities.
                        Lower temperature makes the distribution sharper.
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss.
        
        Args:
            z1: Embeddings from first augmented view, shape (N, D)
            z2: Embeddings from second augmented view, shape (N, D)
            
        Returns:
            Scalar loss value
        """
        batch_size = z1.shape[0]
        device = z1.device
        
        # Normalize embeddings (should already be normalized, but ensure)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        # Concatenate embeddings: [z1_0, z1_1, ..., z2_0, z2_1, ...]
        z = torch.cat([z1, z2], dim=0)  # (2N, D)
        
        # Compute pairwise similarity matrix
        sim = torch.mm(z, z.T) / self.temperature  # (2N, 2N)
        
        # Create mask for positive pairs
        # Positive pairs: (i, i+N) and (i+N, i) for i in [0, N)
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # [N...2N-1, 0...N-1]
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask, float('-inf'))
        
        # Compute loss
        loss = F.cross_entropy(sim, labels)
        
        return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    
    Extends NT-Xent to use label information. Samples with the same label
    are treated as positives, while samples with different labels are negatives.
    
    Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
    ):
        """
        Args:
            temperature: Temperature for scaling similarities
            base_temperature: Base temperature for normalization
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Embeddings of shape (N, D) or (2N, D) if using two views
            labels: Ground truth labels of shape (N,) or (2N,)
            mask: Optional mask of shape (N, N) indicating positive pairs
            
        Returns:
            Scalar loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.mm(features, features.T) / self.temperature  # (N, N)
        
        # Create label mask: 1 if same label, 0 otherwise
        labels = labels.contiguous().view(-1, 1)
        if mask is None:
            mask = torch.eq(labels, labels.T).float().to(device)  # (N, N)
        
        # Mask out self-contrast (diagonal)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        
        # Compute log-prob
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positives
        # Avoid division by zero
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
        
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss with optional hard negative mining.
    
    Pushes anchor closer to positive and farther from negative.
    
    L = max(0, d(a, p) - d(a, n) + margin)
    """
    
    def __init__(
        self,
        margin: float = 0.3,
        distance: str = "cosine",
        mining: str = "hard",
    ):
        """
        Args:
            margin: Margin for triplet loss
            distance: Distance metric ("cosine" or "euclidean")
            mining: Mining strategy ("hard", "semi-hard", "all")
        """
        super().__init__()
        self.margin = margin
        self.distance = distance
        self.mining = mining
    
    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance matrix."""
        if self.distance == "cosine":
            # Cosine distance: 1 - cosine_similarity
            embeddings = F.normalize(embeddings, p=2, dim=1)
            sim = torch.mm(embeddings, embeddings.T)
            return 1 - sim
        else:
            # Euclidean distance
            dot_product = torch.mm(embeddings, embeddings.T)
            square_norm = torch.diag(dot_product)
            distances = square_norm.unsqueeze(0) - 2 * dot_product + square_norm.unsqueeze(1)
            return torch.sqrt(torch.clamp(distances, min=1e-8))
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss for explicit triplets.
        
        Args:
            anchor: Anchor embeddings (N, D)
            positive: Positive embeddings (N, D)
            negative: Negative embeddings (N, D)
            
        Returns:
            Scalar loss value
        """
        if self.distance == "cosine":
            anchor = F.normalize(anchor, p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            negative = F.normalize(negative, p=2, dim=1)
            
            pos_dist = 1 - (anchor * positive).sum(dim=1)
            neg_dist = 1 - (anchor * negative).sum(dim=1)
        else:
            pos_dist = torch.sqrt(((anchor - positive) ** 2).sum(dim=1) + 1e-8)
            neg_dist = torch.sqrt(((anchor - negative) ** 2).sum(dim=1) + 1e-8)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
    
    def forward_with_labels(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss with automatic triplet mining.
        
        Args:
            embeddings: All embeddings (N, D)
            labels: Labels for each embedding (N,)
            
        Returns:
            Scalar loss value
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        # Compute pairwise distances
        pairwise_dist = self._pairwise_distances(embeddings)
        
        # Create masks
        labels = labels.unsqueeze(0)
        same_identity_mask = (labels == labels.T).float()
        not_same_identity_mask = 1 - same_identity_mask
        
        # Remove diagonal
        same_identity_mask = same_identity_mask - torch.eye(batch_size, device=device)
        
        if self.mining == "hard":
            # For each anchor, find hardest positive and hardest negative
            
            # Hardest positive: max distance with same label
            masked_pos = pairwise_dist * same_identity_mask
            hardest_pos, _ = masked_pos.max(dim=1)
            
            # Hardest negative: min distance with different label
            # Add large value to same-identity pairs
            masked_neg = pairwise_dist + same_identity_mask * 1e9
            hardest_neg, _ = masked_neg.min(dim=1)
            
            loss = F.relu(hardest_pos - hardest_neg + self.margin)
            return loss.mean()
        
        elif self.mining == "all":
            # All valid triplets
            pos_dist = pairwise_dist.unsqueeze(2)  # (N, N, 1)
            neg_dist = pairwise_dist.unsqueeze(1)  # (N, 1, N)
            
            # Valid triplet mask: anchor != positive and anchor != negative and pos != neg
            anchor_positive_mask = same_identity_mask.unsqueeze(2)  # (N, N, 1)
            anchor_negative_mask = not_same_identity_mask.unsqueeze(1)  # (N, 1, N)
            triplet_mask = anchor_positive_mask * anchor_negative_mask
            
            # Compute triplet loss
            triplet_loss = pos_dist - neg_dist + self.margin
            triplet_loss = triplet_loss * triplet_mask
            triplet_loss = F.relu(triplet_loss)
            
            # Average only over valid triplets
            num_valid = triplet_mask.sum()
            if num_valid > 0:
                return triplet_loss.sum() / num_valid
            else:
                return torch.tensor(0.0, device=device)
        
        else:
            raise ValueError(f"Unknown mining strategy: {self.mining}")


class CombinedContrastiveLoss(nn.Module):
    """
    Combined loss for flexibility.
    
    Can use NT-Xent, SupCon, or both depending on whether labels are provided.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        use_labels: bool = True,
        supcon_weight: float = 1.0,
        ntxent_weight: float = 0.0,
    ):
        """
        Args:
            temperature: Temperature for contrastive losses
            use_labels: Whether to use labels (SupCon) when available
            supcon_weight: Weight for supervised contrastive loss
            ntxent_weight: Weight for self-supervised NT-Xent loss
        """
        super().__init__()
        self.use_labels = use_labels
        self.supcon_weight = supcon_weight
        self.ntxent_weight = ntxent_weight
        
        self.ntxent = NTXentLoss(temperature)
        self.supcon = SupConLoss(temperature)
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            z1: First view embeddings (N, D)
            z2: Second view embeddings (N, D)
            labels: Optional labels (N,)
            
        Returns:
            Combined loss
        """
        total_loss = 0.0
        
        # NT-Xent (self-supervised)
        if self.ntxent_weight > 0:
            ntxent_loss = self.ntxent(z1, z2)
            total_loss = total_loss + self.ntxent_weight * ntxent_loss
        
        # SupCon (supervised)
        if self.use_labels and labels is not None and self.supcon_weight > 0:
            # Concatenate views and labels
            features = torch.cat([z1, z2], dim=0)
            labels_doubled = torch.cat([labels, labels], dim=0)
            
            supcon_loss = self.supcon(features, labels_doubled)
            total_loss = total_loss + self.supcon_weight * supcon_loss
        elif self.ntxent_weight == 0:
            # If no NT-Xent and no labels, use NT-Xent anyway
            total_loss = self.ntxent(z1, z2)
        
        return total_loss
