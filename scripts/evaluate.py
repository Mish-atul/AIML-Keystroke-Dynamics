#!/usr/bin/env python
"""
Evaluate CLI
============
Evaluate encoder, adapters, and compare with baselines.

Usage:
    python scripts/evaluate.py --encoder artifacts/encoder.pt --split user_disjoint --out results/
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from contrastive_encoder.config import get_default_config, PathConfig
from contrastive_encoder.data.preprocessing import prepare_sequences, create_splits
from contrastive_encoder.models.encoder import KeystrokeEncoder
from contrastive_encoder.models.adapter import UserTemplate
from contrastive_encoder.evaluation.evaluate import (
    evaluate_encoder_identification,
    evaluate_encoder_verification,
    evaluate_with_adapters,
    evaluate_baseline,
    load_baseline_model,
    build_comparison_table,
)
from contrastive_encoder.evaluation.visualization import (
    plot_roc_curves,
    plot_embeddings_tsne,
    plot_score_distributions,
    plot_per_user_eer,
)
from contrastive_encoder.utils.helpers import set_seed, get_device, save_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--encoder",
        type=str,
        default="artifacts/encoder.pt",
        help="Path to encoder",
    )
    parser.add_argument(
        "--adapters-dir",
        type=str,
        default="artifacts/adapters",
        help="Directory with adapter files",
    )
    parser.add_argument(
        "--templates-dir",
        type=str,
        default="artifacts/templates",
        help="Directory with template files",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="cmu_keystroke.csv",
        help="Dataset path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["user_disjoint", "temporal", "both"],
        help="Split strategy",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results",
        help="Output directory",
    )
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="Include baseline comparisons",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (skip visualizations)",
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
        help="Force CPU",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("EVALUATION")
    print("="*60)
    
    set_seed(args.seed)
    device = get_device(prefer_gpu=not args.cpu)
    
    # Load data
    print(f"\nLoading data from: {args.csv}")
    data = prepare_sequences(args.csv, return_dataframe=True)
    
    # Load encoder if exists
    encoder = None
    if os.path.exists(args.encoder):
        print(f"Loading encoder from: {args.encoder}")
        encoder = KeystrokeEncoder()
        encoder.load_state_dict(torch.load(args.encoder, map_location=device))
        encoder.to(device)
        encoder.eval()
    
    # Load templates if exist
    templates = {}
    if os.path.exists(args.templates_dir):
        import json
        for f in os.listdir(args.templates_dir):
            if f.startswith("template_") and f.endswith(".json"):
                user_id = f.replace("template_", "").replace(".json", "")
                template = UserTemplate.load(
                    user_id, args.templates_dir, args.adapters_dir
                )
                templates[user_id] = template
        print(f"Loaded {len(templates)} user templates")
    
    # Splits to evaluate
    splits_to_run = []
    if args.split == "both":
        splits_to_run = ["user_disjoint", "temporal"]
    else:
        splits_to_run = [args.split]
    
    for split_type in splits_to_run:
        print(f"\n{'='*60}")
        print(f"EVALUATING: {split_type.upper()} SPLIT")
        print(f"{'='*60}")
        
        # Create output directory
        out_dir = os.path.join(args.out, split_type)
        os.makedirs(out_dir, exist_ok=True)
        
        # Create splits
        splits = create_splits(data, split_type, random_seed=args.seed)
        
        all_results = {}
        
        # Evaluate encoder (if available)
        if encoder is not None:
            print("\n--- Encoder Evaluation ---")
            
            # Identification
            id_results = evaluate_encoder_identification(
                encoder,
                splits["test"]["X_seq"],
                splits["test"]["y"],
                splits["train"]["X_seq"],
                splits["train"]["y"],
                device,
            )
            print(f"Identification - Top-1: {id_results['top1_accuracy']:.4f}, "
                  f"Top-5: {id_results['top5_accuracy']:.4f}")
            
            # Verification
            ver_results = evaluate_encoder_verification(
                encoder,
                np.concatenate([splits["train"]["X_seq"], splits["test"]["X_seq"]]),
                np.concatenate([splits["train"]["y"], splits["test"]["y"]]),
                device,
            )
            print(f"Verification - AUC: {ver_results['auc']:.4f}, EER: {ver_results['eer']:.4f}")
            
            all_results["Encoder (no adapter)"] = {
                **id_results,
                **ver_results,
            }
            
            # Encoder + Adapters (if templates available)
            if templates:
                print("\n--- Encoder + Adapter Evaluation ---")
                adapter_results = evaluate_with_adapters(
                    encoder,
                    templates,
                    splits["test"]["X_seq"],
                    splits["test"]["y"],
                    data["label_encoder"],
                    device,
                )
                print(f"Encoder+Adapter - AUC: {adapter_results['auc']:.4f}, "
                      f"EER: {adapter_results['eer']:.4f}")
                
                all_results["Encoder + Adapter"] = adapter_results
                
                # Per-user plot
                if not args.quick and adapter_results.get("per_user"):
                    plot_per_user_eer(
                        adapter_results["per_user"],
                        os.path.join(out_dir, "per_user_eer.png"),
                    )
            
            # t-SNE visualization
            if not args.quick:
                print("\nGenerating t-SNE visualization...")
                with torch.no_grad():
                    X_t = torch.FloatTensor(splits["test"]["X_seq"][:2000]).to(device)
                    embeddings = encoder(X_t, normalize=True).cpu().numpy()
                
                plot_embeddings_tsne(
                    embeddings,
                    splits["test"]["y"][:2000],
                    os.path.join(out_dir, "embeddings_tsne.png"),
                )
        
        # Evaluate baselines
        if args.baselines:
            print("\n--- Baseline Evaluation ---")
            paths = PathConfig()
            
            baselines = [
                ("Random Forest", "rf", paths.legacy_rf_model),
                ("XGBoost", "xgb", paths.legacy_xgb_model),
                ("HGBT", "hgb", paths.legacy_hgb_model),
            ]
            
            # Load scaler for baselines
            import pickle
            scaler = None
            if os.path.exists(paths.legacy_scaler):
                with open(paths.legacy_scaler, 'rb') as f:
                    scaler = pickle.load(f)
            
            for name, model_type, model_path in baselines:
                if os.path.exists(model_path):
                    try:
                        model = load_baseline_model(model_type, model_path)
                        results = evaluate_baseline(
                            model, model_type,
                            splits["test"]["X_agg"],
                            splits["test"]["y"],
                            scaler,
                        )
                        print(f"{name}: Acc={results['accuracy']:.4f}, "
                              f"AUC={results['auc']:.4f}, EER={results['eer']:.4f}")
                        all_results[name] = results
                    except Exception as e:
                        print(f"Error evaluating {name}: {e}")
                else:
                    print(f"Baseline not found: {model_path}")
        
        # Save results
        save_json(all_results, os.path.join(out_dir, "evaluation_results.json"))
        
        # Build comparison table
        build_comparison_table(
            all_results,
            os.path.join(out_dir, "model_comparison_table.csv"),
        )
        
        # ROC plot
        if not args.quick:
            plot_roc_curves(
                all_results,
                os.path.join(out_dir, "verification_roc_all_models.png"),
            )
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {args.out}/")


if __name__ == "__main__":
    main()
