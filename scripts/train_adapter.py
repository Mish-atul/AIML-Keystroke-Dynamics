#!/usr/bin/env python
"""
Train Adapter CLI
=================
Command-line interface for training per-user adapters.

Usage:
    python scripts/train_adapter.py --encoder artifacts/encoder.pt --user s002 --shots 5

For all users:
    python scripts/train_adapter.py --encoder artifacts/encoder.pt --all --shots 5
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrastive_encoder.config import get_default_config, AdapterConfig
from contrastive_encoder.data.preprocessing import prepare_sequences
from contrastive_encoder.training.train_adapter import (
    train_adapter_for_user,
    train_adapters_for_all_users,
)
from contrastive_encoder.utils.helpers import set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train per-user adapters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Path to pretrained encoder.pt",
    )
    
    # User selection
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="Single user ID to train adapter for (e.g., s002)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train adapters for all users",
    )
    
    # Data
    parser.add_argument(
        "--csv",
        type=str,
        default="cmu_keystroke.csv",
        help="Path to CSV dataset",
    )
    
    # Training
    parser.add_argument(
        "--shots",
        type=int,
        default=5,
        help="Number of enrollment shots",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Adapter hidden dimension",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum training steps",
    )
    
    # Output
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts",
        help="Output directory for adapters and templates",
    )
    
    # Misc
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


def main():
    args = parse_args()
    
    if not args.user and not args.all:
        print("ERROR: Must specify either --user or --all")
        sys.exit(1)
    
    print("="*60)
    print("ADAPTER TRAINING")
    print("="*60)
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(prefer_gpu=not args.cpu)
    
    # Check encoder exists
    if not os.path.exists(args.encoder):
        print(f"ERROR: Encoder not found at {args.encoder}")
        sys.exit(1)
    
    # Load data
    print(f"\nLoading data from: {args.csv}")
    data = prepare_sequences(args.csv)
    
    # Build config
    adapter_config = AdapterConfig(
        hidden_dim=args.hidden_dim,
        max_steps=args.max_steps,
        default_shots=args.shots,
    )
    
    if args.all:
        # Train for all users
        templates = train_adapters_for_all_users(
            encoder_path=args.encoder,
            data=data,
            n_shots=args.shots,
            output_dir=args.out,
            config=adapter_config,
            device=device,
            seed=args.seed,
        )
        print(f"\nTrained {len(templates)} adapters")
        
    else:
        # Train for single user
        user_id = args.user
        
        # Find user's data
        X_seq = data["X_seq_scaled"]
        y = data["y"]
        label_encoder = data["label_encoder"]
        
        # Find user label
        try:
            # Try to find by subject ID string
            user_label = label_encoder.transform([user_id])[0]
        except ValueError:
            # Try numeric
            try:
                user_label = int(user_id.replace("s", ""))
            except:
                print(f"ERROR: User {user_id} not found")
                sys.exit(1)
        
        user_mask = y == user_label
        user_indices = np.where(user_mask)[0]
        
        if len(user_indices) < args.shots:
            print(f"ERROR: User {user_id} has only {len(user_indices)} samples")
            sys.exit(1)
        
        # Sample enrollment
        np.random.seed(args.seed)
        np.random.shuffle(user_indices)
        X_support = X_seq[user_indices[:args.shots]]
        
        # Get negatives
        neg_mask = ~user_mask
        neg_indices = np.where(neg_mask)[0]
        np.random.shuffle(neg_indices)
        X_negatives = X_seq[neg_indices[:500]]
        
        # Train
        template = train_adapter_for_user(
            encoder_path=args.encoder,
            user_id=user_id,
            X_support=X_support,
            X_negatives=X_negatives,
            output_dir=args.out,
            config=adapter_config,
            device=device,
        )
        
        print(f"\nAdapter saved to: {args.out}/adapters/adapter_{user_id}.pt")
        print(f"Template saved to: {args.out}/templates/template_{user_id}.json")
    
    print("\n" + "="*60)
    print("ADAPTER TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
