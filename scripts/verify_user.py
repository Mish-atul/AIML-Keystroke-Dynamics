#!/usr/bin/env python
"""
Verify User CLI
===============
Verify a keystroke sample against a user's template.

Usage:
    python scripts/verify_user.py --user alice --sample sample.csv
    python scripts/verify_user.py --user alice --sample-json '{"H.period": 0.1, ...}'
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from contrastive_encoder.models.encoder import KeystrokeEncoder
from contrastive_encoder.models.adapter import UserAdapter, UserTemplate, verify_sample
from contrastive_encoder.data.preprocessing import KEYS
from contrastive_encoder.utils.helpers import get_device, load_json
import pickle


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify keystroke sample against user template",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--user",
        type=str,
        required=True,
        help="User ID to verify against",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Path to CSV with single sample (or multiple for batch)",
    )
    parser.add_argument(
        "--sample-json",
        type=str,
        default=None,
        help="JSON string with timing features",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="artifacts/encoder.pt",
        help="Path to encoder",
    )
    parser.add_argument(
        "--templates-dir",
        type=str,
        default="artifacts/templates",
        help="Templates directory",
    )
    parser.add_argument(
        "--adapters-dir",
        type=str,
        default="artifacts/adapters",
        help="Adapters directory",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="artifacts/scaler_seq.pkl",
        help="Path to sequence scaler",
    )
    parser.add_argument(
        "--threshold-override",
        type=float,
        default=None,
        help="Override stored threshold",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def load_sample_from_csv(csv_path: str, scaler_path: str) -> np.ndarray:
    """Load single sample from CSV."""
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
    
    # Apply scaler
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        X_flat = X_seq.reshape(n_samples, -1)
        X_scaled = scaler.transform(X_flat).reshape(n_samples, 11, 3)
        return X_scaled
    
    return X_seq


def load_sample_from_json(json_str: str, scaler_path: str) -> np.ndarray:
    """Load single sample from JSON."""
    data = json.loads(json_str)
    
    X_seq = np.zeros((1, 11, 3), dtype=np.float32)
    
    for i, key in enumerate(KEYS):
        X_seq[0, i, 0] = data.get(f"H.{key}", 0.0)
        
        if i == 0:
            X_seq[0, i, 1] = 0.0
            X_seq[0, i, 2] = 0.0
        else:
            prev_key = KEYS[i - 1]
            X_seq[0, i, 1] = data.get(f"DD.{prev_key}.{key}", 0.0)
            X_seq[0, i, 2] = data.get(f"UD.{prev_key}.{key}", 0.0)
    
    # Apply scaler
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        X_flat = X_seq.reshape(1, -1)
        X_scaled = scaler.transform(X_flat).reshape(1, 11, 3)
        return X_scaled
    
    return X_seq


def main():
    args = parse_args()
    
    device = get_device(prefer_gpu=not args.cpu)
    
    # Check paths
    if not os.path.exists(args.encoder):
        print(f"ERROR: Encoder not found: {args.encoder}")
        sys.exit(1)
    
    template_path = os.path.join(args.templates_dir, f"template_{args.user}.json")
    if not os.path.exists(template_path):
        print(f"ERROR: Template not found: {template_path}")
        sys.exit(1)
    
    adapter_path = os.path.join(args.adapters_dir, f"adapter_{args.user}.pt")
    if not os.path.exists(adapter_path):
        print(f"ERROR: Adapter not found: {adapter_path}")
        sys.exit(1)
    
    # Load sample
    if args.sample:
        samples = load_sample_from_csv(args.sample, args.scaler)
    elif args.sample_json:
        samples = load_sample_from_json(args.sample_json, args.scaler)
    else:
        print("ERROR: Must provide --sample or --sample-json")
        sys.exit(1)
    
    # Load encoder
    encoder = KeystrokeEncoder()
    encoder.load_state_dict(torch.load(args.encoder, map_location=device))
    encoder.to(device)
    encoder.eval()
    
    # Load template
    template_data = load_json(template_path)
    centroid = torch.tensor(template_data["centroid"])
    threshold = args.threshold_override or template_data["threshold"]
    
    # Load adapter
    adapter = UserAdapter()
    adapter.load_state_dict(torch.load(adapter_path, map_location=device))
    adapter.to(device)
    adapter.eval()
    
    # Verify each sample
    print(f"\n{'='*60}")
    print(f"VERIFICATION: {args.user}")
    print(f"{'='*60}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Samples: {len(samples)}")
    print()
    
    results = []
    
    for i, sample in enumerate(samples):
        sample_t = torch.FloatTensor(sample)
        
        result = verify_sample(
            encoder=encoder,
            adapter=adapter,
            sample=sample_t,
            centroid=centroid,
            threshold=threshold,
            device=device,
        )
        
        results.append(result)
        
        status = "ACCEPT ✓" if result["accept"] else "REJECT ✗"
        print(f"Sample {i+1}: {status} (similarity: {result['similarity']:.4f})")
    
    # Summary
    n_accepted = sum(1 for r in results if r["accept"])
    print(f"\nSummary: {n_accepted}/{len(results)} accepted")
    
    # Return code for scripting
    if len(results) == 1:
        sys.exit(0 if results[0]["accept"] else 1)


if __name__ == "__main__":
    main()
