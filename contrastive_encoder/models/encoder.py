"""
1D-CNN Keystroke Encoder
========================
Global encoder architecture for keystroke timing sequences.

Produces 128-d L2-normalized embeddings.

Architecture:
    Input: (batch, 11, 3) - 11 keys, 3 features each
    Conv1D(64, k=3) → BatchNorm → ReLU → MaxPool
    Conv1D(128, k=3) → BatchNorm → ReLU → MaxPool
    Conv1D(256, k=3) → BatchNorm → ReLU → GlobalAvgPool
    Linear(256 → 128) → BatchNorm → ReLU → L2-normalize
    Output: (batch, 128) unit-norm embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class Conv1DBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batch_norm: bool = True,
        pool_size: Optional[int] = None,
    ):
        super().__init__()
        
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
        ]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        if pool_size is not None:
            layers.append(nn.MaxPool1d(pool_size))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class KeystrokeEncoder(nn.Module):
    """
    1D-CNN Encoder for keystroke sequences.
    
    Input shape: (batch, seq_len, features) = (batch, 11, 3)
    Output shape: (batch, embedding_dim) = (batch, 128)
    
    Embeddings are L2-normalized to lie on the unit hypersphere.
    """
    
    def __init__(
        self,
        input_features: int = 3,
        seq_length: int = 11,
        embedding_dim: int = 128,
        conv_channels: List[int] = None,
        kernel_size: int = 3,
        use_batch_norm: bool = True,
    ):
        """
        Args:
            input_features: Number of features per timestep (3 for [H, DD, UD])
            seq_length: Sequence length (11 for the password)
            embedding_dim: Output embedding dimension
            conv_channels: List of channel sizes for conv layers
            kernel_size: Kernel size for conv layers
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        if conv_channels is None:
            conv_channels = [64, 128, 256]
        
        self.input_features = input_features
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        
        # Build convolutional layers
        # Input: (batch, features, seq_len) after transpose
        # Note: Conv1d expects (batch, channels, length)
        
        conv_layers = []
        in_ch = input_features
        
        for i, out_ch in enumerate(conv_channels):
            # Add pooling after first two conv blocks
            pool = 2 if i < 2 else None
            
            conv_layers.append(
                Conv1DBlock(
                    in_ch, out_ch,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    use_batch_norm=use_batch_norm,
                    pool_size=pool,
                )
            )
            in_ch = out_ch
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(conv_channels[-1], embedding_dim),
            nn.BatchNorm1d(embedding_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        
        # Calculate total parameters
        self._param_count = sum(p.numel() for p in self.parameters())
    
    def forward(
        self,
        x: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features) = (batch, 11, 3)
            normalize: Whether to L2-normalize the output
            
        Returns:
            Embeddings of shape (batch, embedding_dim)
        """
        # Transpose for Conv1d: (batch, seq, feat) -> (batch, feat, seq)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Global average pooling: (batch, channels, length) -> (batch, channels, 1)
        x = self.global_pool(x)
        
        # Flatten: (batch, channels, 1) -> (batch, channels)
        x = x.squeeze(-1)
        
        # Projection to embedding dimension
        x = self.projection(x)
        
        # L2 normalize
        if normalize:
            x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return self._param_count
    
    def __repr__(self) -> str:
        return (
            f"KeystrokeEncoder(\n"
            f"  input_features={self.input_features},\n"
            f"  seq_length={self.seq_length},\n"
            f"  embedding_dim={self.embedding_dim},\n"
            f"  parameters={self._param_count:,}\n"
            f")"
        )


class ProjectionHead(nn.Module):
    """
    Optional projection head for contrastive learning.
    
    Maps embeddings to a space where contrastive loss is computed.
    Can be discarded after pretraining.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
    ):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        return F.normalize(x, p=2, dim=1)


class EncoderWithProjection(nn.Module):
    """
    Encoder with optional projection head for contrastive pretraining.
    
    During pretraining, use the full model.
    After pretraining, discard the projection head and use encoder only.
    """
    
    def __init__(
        self,
        encoder: KeystrokeEncoder,
        projection_head: Optional[ProjectionHead] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head
    
    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            return_embedding: If True, return encoder output (for downstream tasks)
                            If False, return projection output (for contrastive loss)
        """
        embedding = self.encoder(x, normalize=True)
        
        if return_embedding or self.projection_head is None:
            return embedding
        
        return self.projection_head(embedding)


def create_encoder(
    input_features: int = 3,
    seq_length: int = 11,
    embedding_dim: int = 128,
    conv_channels: List[int] = None,
    use_projection: bool = True,
    projection_dim: int = 128,
) -> nn.Module:
    """
    Factory function to create encoder.
    
    Args:
        input_features: Features per timestep
        seq_length: Sequence length
        embedding_dim: Encoder embedding dimension
        conv_channels: Conv layer channels
        use_projection: Whether to add projection head for contrastive learning
        projection_dim: Projection head output dimension
        
    Returns:
        Encoder model (with or without projection head)
    """
    encoder = KeystrokeEncoder(
        input_features=input_features,
        seq_length=seq_length,
        embedding_dim=embedding_dim,
        conv_channels=conv_channels,
    )
    
    if use_projection:
        projection = ProjectionHead(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim * 2,
            output_dim=projection_dim,
        )
        return EncoderWithProjection(encoder, projection)
    
    return encoder


# Alias for backward compatibility
Encoder = KeystrokeEncoder
