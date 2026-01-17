#!/usr/bin/env python
"""
Enroll User CLI
===============
Enroll a new user with keystroke samples.

Usage:
    python scripts/enroll_user.py --encoder artifacts/encoder.pt --user alice --samples samples.csv
    python scripts/enroll_user.py --encoder artifacts/encoder.pt --user alice --samples-dir ./enrollment/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrastive_encoder.config import AdapterConfig
from contrastive_encoder.data.preprocessing import prepare_sequences, HOLD_COLUMNS, DD_COLUMNS, UD_COLUMNS, KEYS
from contrastive_encoder.training.train_adapter import train_adapter_for_user
from contrastive_encoder.utils.helpers import set_seed, get_device, load_json
import pickle


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enroll a new user",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Path to pretrained encoder.pt",
    )
    parser.add_argument(
        "--user",
        type=str,
        required=True,
        help="User ID for enrollment",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default=None,
        help="Path to CSV with enrollment samples",
    )
    parser.add_argument(
        "--negatives-csv",
        type=str,
        default="cmu_keystroke.csv",
        help="CSV with negative samples (for training)",
    )
    parser.add_argument(
        "--n-negatives",
        type=int,
        default=500,
        help="Number of negative samples to use",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode",
    )
    
    return parser.parse_args()


def load_enrollment_samples(csv_path: str, scaler_path: str = "artifacts/scaler_seq.pkl") -> np.ndarray:
    """Load and preprocess enrollment samples from CSV."""
    df = pd.read_csv(csv_path)
    
    n_samples = len(df)
    X_seq = np.zeros((n_samples, 11, 3), dtype=np.float32)
    
    for i, key in enumerate(KEYS):
        X_seq[:, i, 0] = df[f"H.{key}"].values
        
        if i == 0:
            X_seq[:, i, 1] = 0.0
            X_seq[:, i, 2] = 0.0
        else:
            prev_key = KEYS[i - 1]
            X_seq[:, i, 1] = df[f"DD.{prev_key}.{key}"].values
            X_seq[:, i, 2] = df[f"UD.{prev_key}.{key}"].values
    
    # Apply scaler if available
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        X_flat = X_seq.reshape(n_samples, -1)
        X_scaled = scaler.transform(X_flat).reshape(n_samples, 11, 3)
        return X_scaled
    
    return X_seq


def main():
    args = parse_args()
    
    print("="*60)
    print("USER ENROLLMENT")
    print("="*60)
    
    set_seed(args.seed)
    device = get_device(prefer_gpu=not args.cpu)
    
    # Check encoder
    if not os.path.exists(args.encoder):
        print(f"ERROR: Encoder not found: {args.encoder}")
        sys.exit(1)
    
    # Load enrollment samples
    if args.samples:
        print(f"Loading enrollment samples from: {args.samples}")
        X_support = load_enrollment_samples(args.samples)
        print(f"Loaded {len(X_support)} enrollment samples")
    else:
        print("ERROR: Must provide --samples")
        sys.exit(1)
    
    # Load negative samples
    print(f"Loading negative samples from: {args.negatives_csv}")
    neg_data = prepare_sequences(args.negatives_csv)
    X_neg = neg_data["X_seq_scaled"]
    
    np.random.shuffle(X_neg)
    X_negatives = X_neg[:args.n_negatives]
    print(f"Using {len(X_negatives)} negative samples")
    
    # Train adapter
    config = AdapterConfig()
    template = train_adapter_for_user(
        encoder_path=args.encoder,
        user_id=args.user,
        X_support=X_support,
        X_negatives=X_negatives,
        output_dir=args.out,
        config=config,
        device=device,
    )
    
    print(f"\n{'='*60}")
    print("ENROLLMENT COMPLETE")
    print(f"{'='*60}")
    print(f"User: {args.user}")
    print(f"Enrollment shots: {len(X_support)}")
    print(f"Threshold: {template.threshold:.4f}")
    print(f"Template: {args.out}/templates/template_{args.user}.json")
    print(f"Adapter: {args.out}/adapters/adapter_{args.user}.pt")


if __name__ == "__main__":
    main()
