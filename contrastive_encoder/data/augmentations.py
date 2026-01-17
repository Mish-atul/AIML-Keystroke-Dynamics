"""
Data Augmentations for Contrastive Learning
============================================
Label-preserving augmentations for keystroke timing data.

Augmentations are conservative for fixed-text keystroke data since
the password structure is fixed and patterns are sensitive.
"""

import numpy as np
from typing import Callable, List, Optional, Union
from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    """Configuration for augmentation parameters."""
    jitter_prob: float = 0.5
    jitter_scale: float = 0.03  # ±3%
    timewarp_prob: float = 0.2
    timewarp_scale: float = 0.1
    keydrop_prob: float = 0.02  # Very conservative
    seed: Optional[int] = None


class BaseAugmentation:
    """Base class for augmentations."""
    
    def __init__(self, p: float = 1.0, seed: Optional[int] = None):
        """
        Args:
            p: Probability of applying this augmentation
            seed: Random seed for reproducibility
        """
        self.p = p
        self.rng = np.random.RandomState(seed)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply augmentation with probability p."""
        if self.rng.random() < self.p:
            return self._apply(x)
        return x.copy()
    
    def _apply(self, x: np.ndarray) -> np.ndarray:
        """Implement in subclass."""
        raise NotImplementedError


class GaussianNoise(BaseAugmentation):
    """
    Add Gaussian noise (jitter) to timing values.
    
    This simulates natural variation in typing speed.
    Noise is applied as a percentage of the original value.
    """
    
    def __init__(
        self,
        scale: float = 0.03,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        """
        Args:
            scale: Standard deviation as fraction of value (0.03 = 3%)
            p: Probability of applying
            seed: Random seed
        """
        super().__init__(p, seed)
        self.scale = scale
    
    def _apply(self, x: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise."""
        x = x.copy()
        
        # Generate noise as percentage of each value
        noise = self.rng.randn(*x.shape) * self.scale
        
        # Apply multiplicative noise: x * (1 + noise)
        x_aug = x * (1 + noise)
        
        # Ensure non-negative (timing values can't be negative)
        x_aug = np.maximum(x_aug, 0.0)
        
        return x_aug


class TimeWarp(BaseAugmentation):
    """
    Apply local time warping (stretch/shrink).
    
    This simulates variations in typing rhythm where certain
    key sequences might be faster or slower.
    """
    
    def __init__(
        self,
        scale: float = 0.1,
        n_segments: int = 3,
        p: float = 0.2,
        seed: Optional[int] = None,
    ):
        """
        Args:
            scale: Maximum warp factor (0.1 = ±10%)
            n_segments: Number of segments to warp independently
            p: Probability of applying
            seed: Random seed
        """
        super().__init__(p, seed)
        self.scale = scale
        self.n_segments = n_segments
    
    def _apply(self, x: np.ndarray) -> np.ndarray:
        """Apply time warping."""
        x = x.copy()
        
        # x shape is (seq_len, features) or (11, 3)
        seq_len = x.shape[0]
        
        # Divide sequence into segments
        segment_size = max(1, seq_len // self.n_segments)
        
        for i in range(0, seq_len, segment_size):
            end = min(i + segment_size, seq_len)
            
            # Random warp factor for this segment
            warp = 1.0 + self.rng.uniform(-self.scale, self.scale)
            
            # Apply to all features in this segment
            x[i:end] = x[i:end] * warp
        
        return np.maximum(x, 0.0)


class KeyDropout(BaseAugmentation):
    """
    Simulate missing or corrupted keystroke.
    
    Very conservative for fixed-text since structure is important.
    Replaces dropped key features with interpolated values.
    """
    
    def __init__(
        self,
        p_drop: float = 0.02,
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            p_drop: Probability of dropping each key
            p: Probability of applying augmentation at all
            seed: Random seed
        """
        super().__init__(p, seed)
        self.p_drop = p_drop
    
    def _apply(self, x: np.ndarray) -> np.ndarray:
        """Apply key dropout with interpolation."""
        x = x.copy()
        seq_len = x.shape[0]
        
        # Randomly select keys to drop (but not first or last)
        for i in range(1, seq_len - 1):
            if self.rng.random() < self.p_drop:
                # Interpolate from neighbors
                x[i] = (x[i-1] + x[i+1]) / 2
        
        return x


class TimeShift(BaseAugmentation):
    """
    Apply global time shift (scale all timings).
    
    Simulates consistently faster or slower typing speed.
    """
    
    def __init__(
        self,
        scale_range: tuple = (0.9, 1.1),
        p: float = 0.3,
        seed: Optional[int] = None,
    ):
        """
        Args:
            scale_range: (min, max) scaling factor
            p: Probability of applying
            seed: Random seed
        """
        super().__init__(p, seed)
        self.scale_range = scale_range
    
    def _apply(self, x: np.ndarray) -> np.ndarray:
        """Apply global time scaling."""
        x = x.copy()
        scale = self.rng.uniform(*self.scale_range)
        return x * scale


class Compose:
    """
    Compose multiple augmentations.
    
    Each augmentation is applied independently with its own probability.
    """
    
    def __init__(
        self,
        augmentations: List[BaseAugmentation],
        shuffle: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            augmentations: List of augmentation objects
            shuffle: Whether to randomize order of augmentations
            seed: Random seed for shuffle
        """
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply all augmentations sequentially."""
        result = x.copy()
        
        augs = list(self.augmentations)
        if self.shuffle:
            self.rng.shuffle(augs)
        
        for aug in augs:
            result = aug(result)
        
        return result


def get_contrastive_augmentations(
    config: Optional[AugmentationConfig] = None,
    seed: Optional[int] = None,
) -> Compose:
    """
    Get the standard augmentation pipeline for contrastive learning.
    
    Args:
        config: Augmentation configuration
        seed: Random seed
        
    Returns:
        Composed augmentation pipeline
    """
    if config is None:
        config = AugmentationConfig()
    
    # Use config seed or provided seed
    base_seed = config.seed if config.seed is not None else seed
    
    augmentations = [
        GaussianNoise(
            scale=config.jitter_scale,
            p=config.jitter_prob,
            seed=base_seed,
        ),
        TimeWarp(
            scale=config.timewarp_scale,
            p=config.timewarp_prob,
            seed=base_seed + 1 if base_seed else None,
        ),
        KeyDropout(
            p_drop=config.keydrop_prob,
            p=1.0 if config.keydrop_prob > 0 else 0.0,
            seed=base_seed + 2 if base_seed else None,
        ),
    ]
    
    return Compose(augmentations, shuffle=False, seed=base_seed)


def get_light_augmentations(seed: Optional[int] = None) -> Compose:
    """Get lighter augmentations for fine-tuning or validation."""
    return Compose([
        GaussianNoise(scale=0.01, p=0.3, seed=seed),
    ])


class TwoViewAugmentation:
    """
    Create two augmented views of the same sample for contrastive learning.
    """
    
    def __init__(
        self,
        augmentation: Compose,
        seed: Optional[int] = None,
    ):
        """
        Args:
            augmentation: Base augmentation pipeline
            seed: Random seed
        """
        self.augmentation = augmentation
        self.rng = np.random.RandomState(seed)
    
    def __call__(self, x: np.ndarray) -> tuple:
        """
        Create two augmented views.
        
        Args:
            x: Input sample (11, 3)
            
        Returns:
            Tuple of (view1, view2) - both augmented versions
        """
        view1 = self.augmentation(x)
        view2 = self.augmentation(x)
        return view1, view2
