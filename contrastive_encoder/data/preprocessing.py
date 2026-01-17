"""
Data Preprocessing
==================
Functions for loading, validating, and preparing keystroke sequences.

Canonical Sequence Format:
- Shape: (N, 11, 3) where:
  - N = number of samples
  - 11 = number of key tokens in password ".tie5Roanl"
  - 3 = features per key: [H (hold), DD (down-to-down), UD (up-to-down)]

The CMU dataset columns follow this pattern:
- H.{key} = Hold/dwell time for key
- DD.{key1}.{key2} = Down-to-Down time from key1 to key2
- UD.{key1}.{key2} = Up-to-Down time from key1 to key2
"""

import os
import shutil
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

from ..config import DataConfig, PathConfig, KEY_ORDER


# =============================================================================
# COLUMN MAPPINGS
# =============================================================================

# Keys in order for the password ".tie5Roanl" + Return
KEYS = ["period", "t", "i", "e", "five", "Shift.r", "o", "a", "n", "l", "Return"]

# Hold time columns (11 total)
HOLD_COLUMNS = [f"H.{key}" for key in KEYS]

# Digraph columns (10 pairs)
DIGRAPH_PAIRS = [
    ("period", "t"), ("t", "i"), ("i", "e"), ("e", "five"),
    ("five", "Shift.r"), ("Shift.r", "o"), ("o", "a"),
    ("a", "n"), ("n", "l"), ("l", "Return")
]

DD_COLUMNS = [f"DD.{k1}.{k2}" for k1, k2 in DIGRAPH_PAIRS]
UD_COLUMNS = [f"UD.{k1}.{k2}" for k1, k2 in DIGRAPH_PAIRS]


def load_and_validate_data(
    csv_path: str,
    expected_users: int = 51,
    expected_samples_per_user: int = 400,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and validate the CMU keystroke dataset.
    
    Args:
        csv_path: Path to CSV file
        expected_users: Expected number of unique users
        expected_samples_per_user: Expected samples per user
        
    Returns:
        Tuple of (DataFrame, validation_report)
        
    Raises:
        ValueError: If validation fails critically
    """
    print(f"Loading dataset from: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validation report
    report = {
        "file_path": csv_path,
        "load_timestamp": datetime.now().isoformat(),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "issues": [],
        "warnings": [],
    }
    
    # Check for required columns
    required_cols = ["subject", "sessionIndex", "rep"] + HOLD_COLUMNS + DD_COLUMNS + UD_COLUMNS
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        report["issues"].append(f"Missing columns: {missing_cols}")
        print(f"WARNING: Missing columns: {missing_cols}")
    
    # Check users
    unique_users = df["subject"].nunique()
    report["unique_users"] = unique_users
    
    if unique_users != expected_users:
        report["warnings"].append(
            f"Expected {expected_users} users, found {unique_users}"
        )
    
    # Check samples per user
    samples_per_user = df.groupby("subject").size()
    report["samples_per_user"] = {
        "min": int(samples_per_user.min()),
        "max": int(samples_per_user.max()),
        "mean": float(samples_per_user.mean()),
    }
    
    # Check for NaN values
    nan_counts = df[HOLD_COLUMNS + DD_COLUMNS + UD_COLUMNS].isna().sum()
    total_nans = nan_counts.sum()
    report["nan_values"] = int(total_nans)
    
    if total_nans > 0:
        report["warnings"].append(f"Found {total_nans} NaN values in timing columns")
        print(f"WARNING: Found {total_nans} NaN values")
    
    # Check for outliers (values > 5 std from mean)
    timing_cols = HOLD_COLUMNS + DD_COLUMNS + UD_COLUMNS
    timing_data = df[timing_cols].values
    means = np.mean(timing_data, axis=0)
    stds = np.std(timing_data, axis=0)
    
    outlier_mask = np.abs(timing_data - means) > 5 * stds
    outlier_count = outlier_mask.sum()
    report["outlier_values"] = int(outlier_count)
    
    if outlier_count > 0:
        report["warnings"].append(f"Found {outlier_count} potential outlier values (>5 std)")
    
    # Summary statistics
    report["timing_stats"] = {
        "hold_mean": float(df[HOLD_COLUMNS].values.mean()),
        "hold_std": float(df[HOLD_COLUMNS].values.std()),
        "dd_mean": float(df[DD_COLUMNS].values.mean()),
        "dd_std": float(df[DD_COLUMNS].values.std()),
        "ud_mean": float(df[UD_COLUMNS].values.mean()),
        "ud_std": float(df[UD_COLUMNS].values.std()),
    }
    
    print(f"Loaded {len(df)} samples from {unique_users} users")
    print(f"Validation: {len(report['issues'])} issues, {len(report['warnings'])} warnings")
    
    return df, report


def prepare_sequences(
    csv_path: str,
    config: Optional[DataConfig] = None,
    legacy_scaler_path: Optional[str] = None,
    return_dataframe: bool = False,
) -> Dict[str, Any]:
    """
    Prepare keystroke data in multiple formats.
    
    This is the main preprocessing function that outputs:
    1. X_seq: Canonical sequence format (N, 11, 3) for the encoder
    2. X_agg: Aggregated features (N, 47) for baseline comparison
    3. y: Integer labels
    4. user_ids: Original user ID strings
    
    Canonical sequence format (N, 11, 3):
    - Dimension 0 (N): samples
    - Dimension 1 (11): key tokens in password
    - Dimension 2 (3): [H (hold), DD (relative), UD (relative)]
      - For first key, DD and UD are set to 0 (no previous key)
    
    Args:
        csv_path: Path to CSV file
        config: Data configuration (uses defaults if None)
        legacy_scaler_path: Path to existing scaler.pkl to reuse
        return_dataframe: Whether to also return the raw DataFrame
        
    Returns:
        Dictionary with keys:
        - X_seq: (N, 11, 3) sequence array
        - X_agg: (N, 47) aggregated features array
        - y: (N,) integer labels
        - user_ids: (N,) string user IDs
        - scaler_seq: fitted StandardScaler for sequences
        - scaler_agg: fitted StandardScaler for aggregated features
        - label_encoder: fitted LabelEncoder
        - df: raw DataFrame (if return_dataframe=True)
    """
    if config is None:
        config = DataConfig()
    
    # Load and validate data
    df, validation_report = load_and_validate_data(csv_path)
    
    # Extract user IDs and encode labels
    user_ids = df["subject"].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(user_ids)
    
    n_samples = len(df)
    n_users = len(label_encoder.classes_)
    print(f"Encoded {n_users} users into integer labels")
    
    # =========================================================================
    # SEQUENCE FORMAT (N, 11, 3)
    # =========================================================================
    print("Preparing canonical sequence format (N, 11, 3)...")
    
    X_seq = np.zeros((n_samples, 11, 3), dtype=np.float32)
    
    for i, key in enumerate(KEYS):
        # Feature 0: Hold time
        X_seq[:, i, 0] = df[f"H.{key}"].values
        
        # Features 1 & 2: DD and UD (relative to previous key)
        if i == 0:
            # First key has no previous key, set to 0
            X_seq[:, i, 1] = 0.0
            X_seq[:, i, 2] = 0.0
        else:
            prev_key = KEYS[i - 1]
            dd_col = f"DD.{prev_key}.{key}"
            ud_col = f"UD.{prev_key}.{key}"
            X_seq[:, i, 1] = df[dd_col].values
            X_seq[:, i, 2] = df[ud_col].values
    
    print(f"  Sequence shape: {X_seq.shape}")
    
    # =========================================================================
    # AGGREGATED FORMAT (N, 47)
    # =========================================================================
    print("Preparing aggregated features (N, 47)...")
    
    # Raw timing features (31 total)
    raw_cols = HOLD_COLUMNS + DD_COLUMNS + UD_COLUMNS
    X_raw = df[raw_cols].values
    
    # Statistical features (16 total)
    stats = []
    
    # Global statistics
    stats.append(np.mean(X_raw, axis=1))  # 1
    stats.append(np.std(X_raw, axis=1))   # 2
    stats.append(np.min(X_raw, axis=1))   # 3
    stats.append(np.max(X_raw, axis=1))   # 4
    stats.append(np.median(X_raw, axis=1))  # 5
    stats.append(np.percentile(X_raw, 25, axis=1))  # 6
    stats.append(np.percentile(X_raw, 75, axis=1))  # 7
    iqr = np.percentile(X_raw, 75, axis=1) - np.percentile(X_raw, 25, axis=1)
    stats.append(iqr)  # 8
    stats.append(np.max(X_raw, axis=1) - np.min(X_raw, axis=1))  # 9 range
    stats.append(np.std(X_raw, axis=1) / (np.mean(X_raw, axis=1) + 1e-10))  # 10 CV
    
    # Feature-specific statistics
    hold_data = df[HOLD_COLUMNS].values
    dd_data = df[DD_COLUMNS].values
    ud_data = df[UD_COLUMNS].values
    
    stats.append(np.mean(hold_data, axis=1))  # 11 hold mean
    stats.append(np.std(hold_data, axis=1))   # 12 hold std
    stats.append(np.mean(dd_data, axis=1))    # 13 DD mean
    stats.append(np.std(dd_data, axis=1))     # 14 DD std
    stats.append(np.mean(ud_data, axis=1))    # 15 UD mean
    stats.append(np.std(ud_data, axis=1))     # 16 UD std
    
    X_stats = np.column_stack(stats)
    X_agg = np.hstack([X_raw, X_stats]).astype(np.float32)
    
    print(f"  Aggregated shape: {X_agg.shape}")
    
    # =========================================================================
    # SCALING
    # =========================================================================
    print("Fitting scalers...")
    
    # Sequence scaler (fit on flattened then reshape)
    scaler_seq = StandardScaler()
    X_seq_flat = X_seq.reshape(n_samples, -1)
    X_seq_scaled = scaler_seq.fit_transform(X_seq_flat).reshape(n_samples, 11, 3)
    
    # Aggregated scaler
    if legacy_scaler_path and os.path.exists(legacy_scaler_path):
        print(f"  Loading legacy scaler from: {legacy_scaler_path}")
        with open(legacy_scaler_path, 'rb') as f:
            scaler_agg = pickle.load(f)
        # Validate feature count matches
        if scaler_agg.n_features_in_ != X_agg.shape[1]:
            print(f"  WARNING: Legacy scaler has {scaler_agg.n_features_in_} features, "
                  f"but data has {X_agg.shape[1]}. Fitting new scaler.")
            scaler_agg = StandardScaler()
            scaler_agg.fit(X_agg)
    else:
        scaler_agg = StandardScaler()
        scaler_agg.fit(X_agg)
    
    X_agg_scaled = scaler_agg.transform(X_agg)
    
    # =========================================================================
    # RESULT
    # =========================================================================
    result = {
        "X_seq": X_seq,
        "X_seq_scaled": X_seq_scaled,
        "X_agg": X_agg,
        "X_agg_scaled": X_agg_scaled,
        "y": y,
        "user_ids": user_ids,
        "scaler_seq": scaler_seq,
        "scaler_agg": scaler_agg,
        "label_encoder": label_encoder,
        "n_samples": n_samples,
        "n_users": n_users,
        "validation_report": validation_report,
    }
    
    if return_dataframe:
        result["df"] = df
    
    return result


def copy_and_normalize_dataset(
    source_path: str,
    dest_path: str,
    manifest_path: str,
) -> Dict[str, Any]:
    """
    Copy dataset to standardized location and create data manifest.
    
    Args:
        source_path: Original CSV path (DSL-StrongPasswordData.csv)
        dest_path: Destination path (cmu_keystroke.csv)
        manifest_path: Path for data_manifest.json
        
    Returns:
        Data manifest dictionary
    """
    print(f"Copying dataset: {source_path} -> {dest_path}")
    
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source dataset not found: {source_path}")
    
    # Copy file
    shutil.copy2(source_path, dest_path)
    
    # Load and validate
    df, validation_report = load_and_validate_data(dest_path)
    
    # Create manifest
    manifest = {
        "dataset": {
            "name": "CMU Keystroke Dynamics Benchmark",
            "source": "DSL-StrongPasswordData.csv",
            "password": ".tie5Roanl",
            "citation": "Killourhy, K.S., Maxion, R.A. (2009). Comparing anomaly-detection "
                       "algorithms for keystroke dynamics. DSN 2009.",
            "license": "Research use",
        },
        "files": {
            "original": source_path,
            "processed": dest_path,
            "manifest": manifest_path,
        },
        "statistics": {
            "total_samples": len(df),
            "unique_users": df["subject"].nunique(),
            "samples_per_user": int(len(df) / df["subject"].nunique()),
            "sessions_per_user": df.groupby("subject")["sessionIndex"].nunique().iloc[0],
        },
        "columns": {
            "metadata": ["subject", "sessionIndex", "rep"],
            "hold_times": HOLD_COLUMNS,
            "dd_times": DD_COLUMNS,
            "ud_times": UD_COLUMNS,
            "total_timing_features": len(HOLD_COLUMNS) + len(DD_COLUMNS) + len(UD_COLUMNS),
        },
        "canonical_format": {
            "shape": "(N, 11, 3)",
            "description": "11 key tokens, 3 features each: [H, DD, UD]",
            "key_order": KEYS,
        },
        "validation": validation_report,
        "created_at": datetime.now().isoformat(),
        "random_seed": 42,
    }
    
    # Save manifest
    from ..utils.helpers import save_json
    save_json(manifest, manifest_path)
    print(f"Created data manifest: {manifest_path}")
    
    return manifest


def create_splits(
    data: Dict[str, Any],
    split_type: str = "user_disjoint",
    config: Optional[DataConfig] = None,
    random_seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Create train/val/test splits for experiments.
    
    Args:
        data: Output from prepare_sequences()
        split_type: "user_disjoint" or "temporal"
        config: Data configuration
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with train/val/test splits
    """
    if config is None:
        config = DataConfig()
    
    np.random.seed(random_seed)
    
    X_seq = data["X_seq_scaled"]
    X_agg = data["X_agg_scaled"]
    y = data["y"]
    user_ids = data["user_ids"]
    
    if split_type == "user_disjoint":
        return _create_user_disjoint_split(
            X_seq, X_agg, y, user_ids, config, random_seed
        )
    elif split_type == "temporal":
        return _create_temporal_split(
            X_seq, X_agg, y, user_ids, data, config, random_seed
        )
    else:
        raise ValueError(f"Unknown split type: {split_type}")


def _create_user_disjoint_split(
    X_seq: np.ndarray,
    X_agg: np.ndarray,
    y: np.ndarray,
    user_ids: np.ndarray,
    config: DataConfig,
    random_seed: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Create user-disjoint split (train on 70% users, test on unseen users)."""
    print("Creating user-disjoint split...")
    
    unique_users = np.unique(y)
    n_users = len(unique_users)
    
    # Shuffle users
    np.random.shuffle(unique_users)
    
    # Split users
    n_train = int(n_users * config.train_ratio)
    n_val = int(n_users * config.val_ratio)
    
    train_users = unique_users[:n_train]
    val_users = unique_users[n_train:n_train + n_val]
    test_users = unique_users[n_train + n_val:]
    
    print(f"  Train users: {len(train_users)}, Val users: {len(val_users)}, Test users: {len(test_users)}")
    
    # Create masks
    train_mask = np.isin(y, train_users)
    val_mask = np.isin(y, val_users)
    test_mask = np.isin(y, test_users)
    
    return {
        "train": {
            "X_seq": X_seq[train_mask],
            "X_agg": X_agg[train_mask],
            "y": y[train_mask],
            "user_ids": user_ids[train_mask],
            "users": train_users,
        },
        "val": {
            "X_seq": X_seq[val_mask],
            "X_agg": X_agg[val_mask],
            "y": y[val_mask],
            "user_ids": user_ids[val_mask],
            "users": val_users,
        },
        "test": {
            "X_seq": X_seq[test_mask],
            "X_agg": X_agg[test_mask],
            "y": y[test_mask],
            "user_ids": user_ids[test_mask],
            "users": test_users,
        },
        "split_type": "user_disjoint",
        "random_seed": random_seed,
    }


def _create_temporal_split(
    X_seq: np.ndarray,
    X_agg: np.ndarray,
    y: np.ndarray,
    user_ids: np.ndarray,
    data: Dict[str, Any],
    config: DataConfig,
    random_seed: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Create temporal split (earlier sessions for train, later for test)."""
    print("Creating temporal split...")
    
    # Need the DataFrame to access session info
    if "df" not in data:
        raise ValueError("Temporal split requires DataFrame. Use return_dataframe=True in prepare_sequences()")
    
    df = data["df"]
    
    # For each user, use earlier sessions for train, later for test
    train_indices = []
    val_indices = []
    test_indices = []
    
    for user in df["subject"].unique():
        user_mask = df["subject"] == user
        user_indices = np.where(user_mask)[0]
        user_sessions = df.loc[user_mask, "sessionIndex"].values
        
        unique_sessions = np.unique(user_sessions)
        n_sessions = len(unique_sessions)
        
        # Split sessions
        n_train_sess = int(n_sessions * config.train_ratio)
        n_val_sess = int(n_sessions * config.val_ratio)
        
        train_sessions = unique_sessions[:n_train_sess]
        val_sessions = unique_sessions[n_train_sess:n_train_sess + n_val_sess]
        test_sessions = unique_sessions[n_train_sess + n_val_sess:]
        
        for idx, sess in zip(user_indices, user_sessions):
            if sess in train_sessions:
                train_indices.append(idx)
            elif sess in val_sessions:
                val_indices.append(idx)
            else:
                test_indices.append(idx)
    
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    print(f"  Train samples: {len(train_indices)}, Val samples: {len(val_indices)}, Test samples: {len(test_indices)}")
    
    return {
        "train": {
            "X_seq": X_seq[train_indices],
            "X_agg": X_agg[train_indices],
            "y": y[train_indices],
            "user_ids": user_ids[train_indices],
        },
        "val": {
            "X_seq": X_seq[val_indices],
            "X_agg": X_agg[val_indices],
            "y": y[val_indices],
            "user_ids": user_ids[val_indices],
        },
        "test": {
            "X_seq": X_seq[test_indices],
            "X_agg": X_agg[test_indices],
            "y": y[test_indices],
            "user_ids": user_ids[test_indices],
        },
        "split_type": "temporal",
        "random_seed": random_seed,
    }


def save_scalers(
    scaler_seq: StandardScaler,
    scaler_agg: StandardScaler,
    label_encoder: LabelEncoder,
    output_dir: str,
) -> None:
    """Save scalers and label encoder to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "scaler_seq.pkl"), 'wb') as f:
        pickle.dump(scaler_seq, f)
    
    with open(os.path.join(output_dir, "scaler_agg.pkl"), 'wb') as f:
        pickle.dump(scaler_agg, f)
    
    with open(os.path.join(output_dir, "label_encoder.pkl"), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"Saved scalers to: {output_dir}")


def load_scalers(
    input_dir: str,
) -> Tuple[StandardScaler, StandardScaler, LabelEncoder]:
    """Load scalers and label encoder from files."""
    with open(os.path.join(input_dir, "scaler_seq.pkl"), 'rb') as f:
        scaler_seq = pickle.load(f)
    
    with open(os.path.join(input_dir, "scaler_agg.pkl"), 'rb') as f:
        scaler_agg = pickle.load(f)
    
    with open(os.path.join(input_dir, "label_encoder.pkl"), 'rb') as f:
        label_encoder = pickle.load(f)
    
    return scaler_seq, scaler_agg, label_encoder
