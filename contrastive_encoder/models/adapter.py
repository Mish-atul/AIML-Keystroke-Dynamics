"""
Per-User Adapter
================
Small MLP adapter for few-shot personalization.

Adapts the global encoder's embeddings for individual users.
Target: <10k parameters per user.

Architecture:
    Input: 128-d embedding
    Linear(128 → 64) → ReLU → Dropout(0.2)
    Linear(64 → 128)
    Output: 128-d adapted embedding

Parameters: 128*64 + 64 + 64*128 + 128 = 16,576
(Can be reduced with smaller hidden dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import os
import json
from datetime import datetime


class UserAdapter(nn.Module):
    """
    Per-user adapter MLP.
    
    Takes encoder embeddings and adapts them for a specific user.
    Designed to be small (<10k parameters) for efficient storage.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        output_dim: int = 128,
        dropout: float = 0.2,
        use_layer_norm: bool = False,
        residual: bool = True,
    ):
        """
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
            residual: Whether to add residual connection
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.residual = residual and (input_dim == output_dim)
        
        # MLP layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Optional layer norm
        self.layer_norm = nn.LayerNorm(output_dim) if use_layer_norm else None
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation
        self.activation = nn.ReLU(inplace=True)
        
        # Count parameters
        self._param_count = sum(p.numel() for p in self.parameters())
    
    def forward(
        self,
        x: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input embeddings from encoder (batch, input_dim)
            normalize: Whether to L2-normalize output
            
        Returns:
            Adapted embeddings (batch, output_dim)
        """
        identity = x
        
        # MLP
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Residual connection
        if self.residual:
            out = out + identity
        
        # Layer norm
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        
        # L2 normalize
        if normalize:
            out = F.normalize(out, p=2, dim=1)
        
        return out
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return self._param_count
    
    def __repr__(self) -> str:
        return (
            f"UserAdapter(\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  residual={self.residual},\n"
            f"  parameters={self._param_count:,}\n"
            f")"
        )


class UserTemplate:
    """
    User template containing adapter, centroid, and metadata.
    
    Stores all information needed for verification.
    """
    
    def __init__(
        self,
        user_id: str,
        adapter: Optional[UserAdapter] = None,
        centroid: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        enrollment_shots: int = 5,
        encoder_hash: str = "none",
        adapter_version: str = "1.0.0",
    ):
        """
        Args:
            user_id: Unique user identifier
            adapter: Trained UserAdapter model
            centroid: Mean embedding vector
            threshold: Decision threshold for verification
            enrollment_shots: Number of samples used for enrollment
            encoder_hash: Hash of encoder used for training
            adapter_version: Version string for adapter format
        """
        self.user_id = user_id
        self.adapter = adapter
        self.centroid = centroid
        self.threshold = threshold
        self.enrollment_shots = enrollment_shots
        self.encoder_hash = encoder_hash
        self.adapter_version = adapter_version
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "user_id": self.user_id,
            "adapter_path": f"adapter_{self.user_id}.pt",
            "centroid": self.centroid.cpu().tolist() if self.centroid is not None else None,
            "threshold": self.threshold,
            "enrollment_shots": self.enrollment_shots,
            "encoder_hash": self.encoder_hash,
            "adapter_version": self.adapter_version,
            "created_at": self.created_at,
        }
    
    def save(
        self,
        templates_dir: str,
        adapters_dir: str,
    ) -> None:
        """
        Save adapter and template to disk.
        
        Args:
            templates_dir: Directory for template JSON files
            adapters_dir: Directory for adapter .pt files
        """
        os.makedirs(templates_dir, exist_ok=True)
        os.makedirs(adapters_dir, exist_ok=True)
        
        # Save adapter weights
        if self.adapter is not None:
            adapter_path = os.path.join(adapters_dir, f"adapter_{self.user_id}.pt")
            torch.save(self.adapter.state_dict(), adapter_path)
        
        # Save template JSON
        template_path = os.path.join(templates_dir, f"template_{self.user_id}.json")
        with open(template_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(
        cls,
        user_id: str,
        templates_dir: str,
        adapters_dir: str,
        adapter_config: Optional[Dict[str, Any]] = None,
    ) -> "UserTemplate":
        """
        Load adapter and template from disk.
        
        Args:
            user_id: User identifier
            templates_dir: Directory containing template JSON files
            adapters_dir: Directory containing adapter .pt files
            adapter_config: Optional config for adapter architecture
            
        Returns:
            Loaded UserTemplate
        """
        # Load template JSON
        template_path = os.path.join(templates_dir, f"template_{user_id}.json")
        with open(template_path, 'r') as f:
            data = json.load(f)
        
        # Load adapter weights
        adapter_path = os.path.join(adapters_dir, f"adapter_{user_id}.pt")
        
        if adapter_config is None:
            adapter_config = {}
        
        adapter = UserAdapter(**adapter_config)
        adapter.load_state_dict(torch.load(adapter_path, map_location='cpu'))
        
        # Convert centroid back to tensor
        centroid = None
        if data.get("centroid") is not None:
            centroid = torch.tensor(data["centroid"])
        
        template = cls(
            user_id=data["user_id"],
            adapter=adapter,
            centroid=centroid,
            threshold=data.get("threshold", 0.5),
            enrollment_shots=data.get("enrollment_shots", 5),
            encoder_hash=data.get("encoder_hash", "none"),
            adapter_version=data.get("adapter_version", "1.0.0"),
        )
        template.created_at = data.get("created_at", "")
        
        return template


def create_adapter(
    hidden_dim: int = 64,
    dropout: float = 0.2,
    residual: bool = True,
    **kwargs,
) -> UserAdapter:
    """
    Factory function to create user adapter.
    
    Args:
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
        residual: Whether to use residual connection
        
    Returns:
        UserAdapter instance
    """
    return UserAdapter(
        input_dim=128,
        hidden_dim=hidden_dim,
        output_dim=128,
        dropout=dropout,
        residual=residual,
        **kwargs,
    )


def compute_centroid(
    encoder: nn.Module,
    adapter: UserAdapter,
    samples: torch.Tensor,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Compute centroid (mean embedding) for user's enrollment samples.
    
    Args:
        encoder: Trained encoder model
        adapter: User's adapter
        samples: Enrollment samples (k, 11, 3)
        device: Compute device
        
    Returns:
        Centroid tensor (128,)
    """
    encoder.eval()
    adapter.eval()
    
    with torch.no_grad():
        samples = samples.to(device)
        
        # Get encoder embeddings
        embeddings = encoder(samples)
        
        # Apply adapter
        adapted = adapter(embeddings)
        
        # Compute mean
        centroid = adapted.mean(dim=0)
        
        # Normalize
        centroid = F.normalize(centroid, p=2, dim=0)
    
    return centroid


def verify_sample(
    encoder: nn.Module,
    adapter: UserAdapter,
    sample: torch.Tensor,
    centroid: torch.Tensor,
    threshold: float,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, Any]:
    """
    Verify a single sample against user's template.
    
    Args:
        encoder: Trained encoder model
        adapter: User's adapter
        sample: Input sample (11, 3) or (1, 11, 3)
        centroid: User's centroid embedding
        threshold: Decision threshold
        device: Compute device
        
    Returns:
        Dictionary with:
        - accept: bool, whether to accept
        - similarity: float, cosine similarity
        - threshold: float, decision threshold
    """
    encoder.eval()
    adapter.eval()
    
    with torch.no_grad():
        # Ensure batch dimension
        if sample.dim() == 2:
            sample = sample.unsqueeze(0)
        
        sample = sample.to(device)
        centroid = centroid.to(device)
        
        # Get embedding
        embedding = encoder(sample)
        adapted = adapter(embedding)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(adapted, centroid.unsqueeze(0)).item()
        
        # Decision
        accept = similarity >= threshold
    
    return {
        "accept": accept,
        "similarity": similarity,
        "threshold": threshold,
    }
