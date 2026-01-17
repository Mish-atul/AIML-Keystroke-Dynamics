"""
Contrastive Encoder Configuration
=================================
Central configuration for hyperparameters, paths, and constants.
All values are CLI-overridable via argparse.

Canonical Sequence Format:
- Shape: (N, 11, 3) where:
  - N = number of samples
  - 11 = number of key tokens in password ".tie5Roanl"
  - 3 = features per key: [H (hold/dwell), DD (down-to-down), UD (up-to-down)]

References:
- CMU Keystroke Dynamics Benchmark (Killourhy & Maxion)
- SimCLR / NT-Xent (Chen et al. 2020)
- Supervised Contrastive Learning (Khosla et al. 2020)
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
import hashlib


# =============================================================================
# PATHS
# =============================================================================
@dataclass
class PathConfig:
    """File and directory paths"""
    # Data
    raw_csv: str = "DSL-StrongPasswordData.csv"
    processed_csv: str = "cmu_keystroke.csv"
    data_manifest: str = "data_manifest.json"
    
    # Models
    artifacts_dir: str = "artifacts"
    encoder_path: str = "artifacts/encoder.pt"
    encoder_keras_path: str = "artifacts/encoder.keras"
    adapters_dir: str = "artifacts/adapters"
    templates_dir: str = "artifacts/templates"
    checkpoints_dir: str = "artifacts/checkpoints"
    
    # Results
    results_dir: str = "results"
    user_disjoint_dir: str = "results/user_disjoint"
    temporal_split_dir: str = "results/temporal_split"
    
    # Legacy (for baseline comparison)
    legacy_scaler: str = "model training/models/scaler.pkl"
    legacy_label_encoder: str = "model training/models/label_encoder.pkl"
    legacy_rf_model: str = "model training/models/rf_model.pkl"
    legacy_xgb_model: str = "model training/models/xgb_model.json"
    legacy_hgb_model: str = "model training/models/hgb_model.pkl"
    legacy_mlp_model: str = "model training/models/mlp_model.h5"
    legacy_cnn_model: str = "model training/models/cnn_model.h5"


# =============================================================================
# DATA CONFIGURATION
# =============================================================================
@dataclass
class DataConfig:
    """Data processing configuration"""
    # Password structure
    password: str = ".tie5Roanl"
    num_keys: int = 11  # Number of key tokens
    
    # Feature structure (canonical format)
    # 3 features per key: [H (hold), DD (down-to-down), UD (up-to-down)]
    features_per_key: int = 3
    sequence_length: int = 11  # Same as num_keys
    
    # Column prefixes in raw CSV
    hold_prefix: str = "H."      # Dwell/hold times
    dd_prefix: str = "DD."       # Down-to-down (digraph)
    ud_prefix: str = "UD."       # Up-to-down (flight)
    
    # Aggregated features (for baseline comparison)
    num_raw_features: int = 31   # 11 H + 10 DD + 10 UD
    num_stat_features: int = 16  # Statistical aggregates
    total_agg_features: int = 47 # Raw + stats
    
    # Splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Seed for reproducibility
    random_seed: int = 42


# =============================================================================
# ENCODER CONFIGURATION
# =============================================================================
@dataclass
class EncoderConfig:
    """1D-CNN Encoder hyperparameters"""
    # Architecture
    embedding_dim: int = 128
    conv_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    kernel_size: int = 3
    use_batch_norm: bool = True
    
    # Training
    batch_size: int = 128  # Use 256 if GPU & memory available
    epochs: int = 120
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    
    # Contrastive learning
    temperature: float = 0.07  # NT-Xent temperature (τ)
    use_supcon: bool = True    # Use supervised contrastive when labels available
    
    # Augmentation probabilities
    aug_jitter_prob: float = 0.5
    aug_jitter_scale: float = 0.03  # ±3%
    aug_timewarp_prob: float = 0.2
    aug_timewarp_scale: float = 0.1
    aug_keydrop_prob: float = 0.02  # Conservative for fixed-text
    
    # Checkpointing
    checkpoint_every: int = 5
    early_stop_patience: int = 20
    disable_early_stop: bool = False  # Set True for fixed epoch training
    
    # Mixed precision
    use_amp: bool = True  # Automatic mixed precision
    
    # Small batch mode (for CPU/low memory)
    small_batch_mode: bool = False
    small_batch_size: int = 32
    memory_bank_size: int = 4096  # MoCo-style queue


# =============================================================================
# ADAPTER CONFIGURATION
# =============================================================================
@dataclass
class AdapterConfig:
    """Per-user adapter hyperparameters"""
    # Architecture (target: <10k parameters)
    # 128->64: 128*64 + 64 = 8,256 params
    # 64->128: 64*128 + 128 = 8,320 params
    # Total: ~16.5k params (with dropout, still lightweight)
    # Reduced version: 128->64->128 ≈ 8.5k params
    hidden_dim: int = 64
    dropout: float = 0.2
    use_layer_norm: bool = False
    
    # Training
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    max_steps: int = 200
    early_stop_patience: int = 20
    
    # Loss
    loss_type: str = "triplet"  # Options: "triplet", "contrastive", "crossentropy"
    triplet_margin: float = 0.3
    num_negatives: int = 32  # Negative samples for contrastive
    use_hard_negatives: bool = True
    
    # Enrollment
    default_shots: int = 5
    experiment_shots: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    experiment_repeats: int = 5  # Random draws per shot count


# =============================================================================
# VERIFICATION CONFIGURATION
# =============================================================================
@dataclass
class VerificationConfig:
    """Verification/threshold configuration"""
    # Similarity metric
    similarity_metric: str = "cosine"  # Options: "cosine", "euclidean"
    
    # Threshold selection
    threshold_method: str = "eer"  # Options: "eer", "far_target"
    target_far: float = 0.01  # 1% FAR for threshold calibration
    
    # Per-user calibration
    calibration_samples: int = 10  # Validation samples for threshold


# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================
@dataclass
class EvaluationConfig:
    """Evaluation and metrics configuration"""
    # Splits to run
    run_user_disjoint: bool = True
    run_temporal: bool = True
    
    # Metrics
    compute_identification: bool = True
    compute_verification: bool = True
    
    # Visualization
    generate_roc_plots: bool = True
    generate_tsne: bool = True
    generate_umap: bool = True
    generate_confusion_matrix: bool = True
    
    # Per-user analysis
    per_user_metrics: bool = True


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
@dataclass
class LoggingConfig:
    """Logging and experiment tracking"""
    use_wandb: bool = False  # Opt-in
    use_tensorboard: bool = True  # Default
    wandb_project: str = "keystroke-contrastive"
    wandb_entity: Optional[str] = None
    log_dir: str = "logs"
    verbose: bool = True


# =============================================================================
# MASTER CONFIGURATION
# =============================================================================
@dataclass
class Config:
    """Master configuration container"""
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Version tracking
    config_version: str = "1.0.0"
    
    def get_encoder_hash(self, encoder_path: str) -> str:
        """Compute hash of encoder file for versioning"""
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()[:8]
        return "none"


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()


def create_directories(config: Config) -> None:
    """Create necessary directories"""
    dirs = [
        config.paths.artifacts_dir,
        config.paths.adapters_dir,
        config.paths.templates_dir,
        config.paths.checkpoints_dir,
        config.paths.results_dir,
        config.paths.user_disjoint_dir,
        config.paths.temporal_split_dir,
        config.logging.log_dir,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# =============================================================================
# KEY MAPPING (Password: .tie5Roanl)
# =============================================================================
# The password ".tie5Roanl" has 11 characters:
# Index 0: . (period)
# Index 1: t
# Index 2: i
# Index 3: e
# Index 4: 5
# Index 5: R (Shift+r)
# Index 6: o
# Index 7: a
# Index 8: n
# Index 9: l
# Index 10: Return (Enter)

KEY_ORDER = [
    "period",  # .
    "t",
    "i",
    "e",
    "five",    # 5
    "Shift.r", # R (uppercase)
    "o",
    "a",
    "n",
    "l",
    "Return",  # Enter
]

# Column name mappings for the CMU dataset
# H.period = Hold time for period key
# DD.period.t = Down-Down time from period to t
# UD.period.t = Up-Down time from period to t
