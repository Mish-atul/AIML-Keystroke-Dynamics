#!/usr/bin/env python
"""
Train Encoder CLI
=================
Command-line interface for training the global keystroke encoder.

Usage:
    python scripts/train_encoder.py --csv cmu_keystroke.csv --out artifacts/encoder.pt --epochs 120

Options:
    --csv           Path to CSV dataset
    --out           Output path for encoder.pt
    --epochs        Number of training epochs (default: 120)
    --batch-size    Batch size (default: 128)
    --lr            Learning rate (default: 1e-3)
    --temperature   Contrastive temperature (default: 0.07)
    --no-supcon     Disable supervised contrastive (use NT-Xent only)
    --small-batch   Use small batch mode for CPU/low memory
    --use-wandb     Enable Weights & Biases logging
    --seed          Random seed (default: 42)
    --split         Split strategy: user_disjoint, temporal, or both (default: user_disjoint)
"""

import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrastive_encoder.config import (
    get_default_config,
    create_directories,
    EncoderConfig,
)
from contrastive_encoder.data.preprocessing import (
    prepare_sequences,
    create_splits,
    copy_and_normalize_dataset,
    save_scalers,
)
from contrastive_encoder.training.train_encoder import EncoderTrainer
from contrastive_encoder.utils.helpers import set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train keystroke contrastive encoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data
    parser.add_argument(
        "--csv", 
        type=str, 
        default="cmu_keystroke.csv",
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--legacy-scaler",
        type=str,
        default=None,
        help="Path to existing scaler.pkl to reuse",
    )
    
    # Output
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/encoder.pt",
        help="Output path for encoder",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="artifacts/checkpoints",
        help="Directory for checkpoints",
    )
    
    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Contrastive temperature",
    )
    parser.add_argument(
        "--no-supcon",
        action="store_true",
        help="Disable supervised contrastive loss",
    )
    parser.add_argument(
        "--small-batch",
        action="store_true",
        help="Use small batch mode for CPU/low memory",
    )
    
    # Split
    parser.add_argument(
        "--split",
        type=str,
        default="user_disjoint",
        choices=["user_disjoint", "temporal", "both"],
        help="Data split strategy",
    )
    
    # Logging
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Log directory",
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
    
    print("="*60)
    print("KEYSTROKE ENCODER TRAINING")
    print("="*60)
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(prefer_gpu=not args.cpu)
    
    # Create directories
    config = get_default_config()
    create_directories(config)
    
    # Check if dataset exists
    csv_path = args.csv
    if not os.path.exists(csv_path):
        # Try to copy from original location
        original_path = "model training/DSL-StrongPasswordData.csv"
        if os.path.exists(original_path):
            print(f"Copying dataset from {original_path}...")
            copy_and_normalize_dataset(
                original_path,
                csv_path,
                "data_manifest.json",
            )
        else:
            print(f"ERROR: Dataset not found at {csv_path}")
            print(f"Please ensure the dataset exists or copy it manually")
            sys.exit(1)
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    data = prepare_sequences(
        csv_path,
        legacy_scaler_path=args.legacy_scaler,
        return_dataframe=True,
    )
    
    # Save scalers
    save_scalers(
        data["scaler_seq"],
        data["scaler_agg"],
        data["label_encoder"],
        "artifacts",
    )
    
    # Create splits
    splits_to_run = []
    if args.split == "both":
        splits_to_run = ["user_disjoint", "temporal"]
    else:
        splits_to_run = [args.split]
    
    for split_type in splits_to_run:
        print(f"\n{'='*60}")
        print(f"TRAINING WITH {split_type.upper()} SPLIT")
        print(f"{'='*60}")
        
        splits = create_splits(data, split_type=split_type, random_seed=args.seed)
        
        # Build encoder config
        encoder_config = EncoderConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            temperature=args.temperature,
            use_supcon=not args.no_supcon,
            small_batch_mode=args.small_batch,
        )
        
        # Output paths
        if len(splits_to_run) > 1:
            out_path = args.out.replace(".pt", f"_{split_type}.pt")
            checkpoint_dir = os.path.join(args.checkpoint_dir, split_type)
            log_dir = os.path.join(args.log_dir, split_type)
        else:
            out_path = args.out
            checkpoint_dir = args.checkpoint_dir
            log_dir = args.log_dir
        
        # Create trainer
        trainer = EncoderTrainer(
            config=encoder_config,
            device=device,
            use_wandb=args.use_wandb,
            use_tensorboard=True,
            log_dir=log_dir,
        )
        
        # Train
        history = trainer.train(
            train_data=splits["train"],
            val_data=splits["val"],
            checkpoint_dir=checkpoint_dir,
            save_path=out_path,
        )
        
        print(f"\nEncoder saved to: {out_path}")
        print(f"Best loss: {history['best_loss']:.4f} at epoch {history['best_epoch']}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
