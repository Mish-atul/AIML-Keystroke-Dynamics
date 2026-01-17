#!/usr/bin/env python
"""
Comprehensive Evaluation Script
===============================
Compares all models and shot counts for the final evaluation.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from contrastive_encoder.data.preprocessing import prepare_sequences, create_splits
from contrastive_encoder.models.encoder import KeystrokeEncoder
from contrastive_encoder.models.adapter import UserAdapter, UserTemplate
from contrastive_encoder.evaluation.evaluate import (
    evaluate_encoder_identification,
    evaluate_encoder_verification,
    compute_eer,
)
from contrastive_encoder.utils.helpers import set_seed, get_device, save_json, load_json


def evaluate_with_adapters_full(
    encoder, templates_dir, adapters_dir, X_test, y_test, label_encoder, device
):
    """Evaluate encoder + adapters."""
    encoder.eval()
    
    all_genuine_scores = []
    all_impostor_scores = []
    
    # Load all templates
    templates = {}
    if os.path.exists(templates_dir):
        for f in os.listdir(templates_dir):
            if f.startswith("template_") and f.endswith(".json"):
                user_id = f.replace("template_", "").replace(".json", "")
                template_path = os.path.join(templates_dir, f)
                templates[user_id] = load_json(template_path)
    
    if not templates:
        return None
    
    with torch.no_grad():
        for user_id, template_data in templates.items():
            try:
                user_label = label_encoder.transform([user_id])[0]
            except:
                try:
                    user_label = int(user_id.replace("s", ""))
                except:
                    continue
            
            user_mask = y_test == user_label
            other_mask = ~user_mask
            
            if user_mask.sum() == 0:
                continue
            
            # Load centroid
            centroid = torch.tensor(template_data["centroid"]).to(device)
            
            # Load adapter if exists
            adapter_path = os.path.join(adapters_dir, f"adapter_{user_id}.pt")
            adapter = None
            if os.path.exists(adapter_path):
                adapter = UserAdapter()
                adapter.load_state_dict(torch.load(adapter_path, map_location=device))
                adapter.to(device)
                adapter.eval()
            
            # Genuine samples
            X_user = torch.FloatTensor(X_test[user_mask]).to(device)
            emb = encoder(X_user, normalize=True)
            if adapter:
                emb = adapter(emb, normalize=True)
            genuine_scores = F.cosine_similarity(emb, centroid.unsqueeze(0)).cpu().numpy()
            all_genuine_scores.extend(genuine_scores.tolist())
            
            # Impostor samples (subset)
            n_impostors = min(int(user_mask.sum()) * 5, other_mask.sum())
            if n_impostors > 0:
                impostor_indices = np.random.choice(np.where(other_mask)[0], size=n_impostors, replace=False)
                X_impostor = torch.FloatTensor(X_test[impostor_indices]).to(device)
                emb = encoder(X_impostor, normalize=True)
                if adapter:
                    emb = adapter(emb, normalize=True)
                impostor_scores = F.cosine_similarity(emb, centroid.unsqueeze(0)).cpu().numpy()
                all_impostor_scores.extend(impostor_scores.tolist())
    
    if not all_genuine_scores or not all_impostor_scores:
        return None
    
    genuine = np.array(all_genuine_scores)
    impostor = np.array(all_impostor_scores)
    
    y_true = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    y_scores = np.concatenate([genuine, impostor])
    
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    eer, eer_threshold = compute_eer(y_true, y_scores)
    
    return {
        "auc": roc_auc,
        "eer": eer,
        "genuine_mean": float(genuine.mean()),
        "impostor_mean": float(impostor.mean()),
    }


def main():
    set_seed(42)
    device = get_device()
    
    print("="*60)
    print("COMPREHENSIVE EVALUATION")
    print("="*60)
    
    # Load data
    data = prepare_sequences("cmu_keystroke.csv", return_dataframe=True)
    
    # Load encoders
    encoders = {}
    if os.path.exists("artifacts/pilot_encoder.pt"):
        enc = KeystrokeEncoder()
        enc.load_state_dict(torch.load("artifacts/pilot_encoder.pt", map_location=device))
        enc.to(device).eval()
        encoders["Pilot (5 ep)"] = enc
    
    if os.path.exists("artifacts/full_encoder.pt"):
        enc = KeystrokeEncoder()
        enc.load_state_dict(torch.load("artifacts/full_encoder.pt", map_location=device))
        enc.to(device).eval()
        encoders["Full (21 ep)"] = enc
    
    # Results
    all_results = {}
    shot_results = {}  # For EER vs shots plot
    
    for split_type in ["user_disjoint", "temporal"]:
        print(f"\n--- {split_type.upper()} SPLIT ---")
        
        splits = create_splits(data, split_type, random_seed=42)
        X_train = splits["train"]["X_seq"]
        y_train = splits["train"]["y"]
        X_test = splits["test"]["X_seq"]
        y_test = splits["test"]["y"]
        
        split_results = {}
        
        # Baselines
        import pickle
        baselines = [
            ("Random Forest", "model training/models/rf_model.pkl"),
            ("XGBoost", "model training/models/xgb_model.json"),
        ]
        
        for name, path in baselines:
            if os.path.exists(path):
                try:
                    if path.endswith(".pkl"):
                        with open(path, 'rb') as f:
                            model = pickle.load(f)
                    else:
                        import xgboost as xgb
                        model = xgb.XGBClassifier()
                        model.load_model(path)
                    
                    X_test_agg = splits["test"]["X_agg"]
                    scaler_path = "model training/models/scaler.pkl"
                    if os.path.exists(scaler_path):
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)
                        X_test_agg = scaler.transform(X_test_agg)
                    
                    preds = model.predict(X_test_agg)
                    pred_proba = model.predict_proba(X_test_agg)
                    
                    from sklearn.metrics import accuracy_score
                    acc = accuracy_score(y_test, preds)
                    
                    max_proba = np.max(pred_proba, axis=1)
                    is_correct = (preds == y_test).astype(int)
                    eer_val, _ = compute_eer(is_correct, max_proba)
                    
                    from sklearn.metrics import roc_auc_score
                    try:
                        auc_val = roc_auc_score(y_test, pred_proba, multi_class='ovr')
                    except:
                        auc_val = 0.0
                    
                    split_results[name] = {"acc": acc, "auc": auc_val, "eer": eer_val}
                    print(f"  {name}: Acc={acc:.4f}, EER={eer_val:.4f}")
                except Exception as e:
                    print(f"  {name}: Error - {e}")
        
        # Encoders (no adapter)
        for enc_name, encoder in encoders.items():
            ver_results = evaluate_encoder_verification(encoder, X_test, y_test, device)
            split_results[f"{enc_name} (centroid)"] = {
                "auc": ver_results["auc"],
                "eer": ver_results["eer"],
            }
            print(f"  {enc_name} (centroid): AUC={ver_results['auc']:.4f}, EER={ver_results['eer']:.4f}")
        
        # Encoder + Adapters for different k
        if "Full (21 ep)" in encoders:
            encoder = encoders["Full (21 ep)"]
            
            for k in [1, 3, 5, 10]:
                templates_dir = f"artifacts/adapters_k{k}/templates"
                adapters_dir = f"artifacts/adapters_k{k}/adapters"
                
                if os.path.exists(templates_dir):
                    result = evaluate_with_adapters_full(
                        encoder, templates_dir, adapters_dir,
                        X_test, y_test, data["label_encoder"], device
                    )
                    
                    if result:
                        key = f"Full + Adapter (k={k})"
                        split_results[key] = result
                        print(f"  {key}: AUC={result['auc']:.4f}, EER={result['eer']:.4f}")
                        
                        if split_type == "user_disjoint":
                            shot_results[k] = result
        
        all_results[split_type] = split_results
    
    # Save results
    os.makedirs("results/full", exist_ok=True)
    save_json(all_results, "results/full/comprehensive_results.json")
    
    # Build comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE (User-Disjoint Split)")
    print("="*60)
    
    table_data = []
    for model, metrics in all_results["user_disjoint"].items():
        table_data.append({
            "Model": model,
            "AUC": f"{metrics.get('auc', 0):.4f}",
            "EER": f"{metrics.get('eer', 0):.4f}",
            "Accuracy": f"{metrics.get('acc', '-')}",
        })
    
    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))
    df.to_csv("results/full/comparison_table.csv", index=False)
    
    # EER vs Shots plot
    if shot_results:
        print("\n" + "="*60)
        print("EER vs ENROLLMENT SHOTS")
        print("="*60)
        
        shots = sorted(shot_results.keys())
        eers = [shot_results[k]["eer"] for k in shots]
        aucs = [shot_results[k]["auc"] for k in shots]
        
        print(f"Shots: {shots}")
        print(f"EERs:  {[f'{e:.4f}' for e in eers]}")
        print(f"AUCs:  {[f'{a:.4f}' for a in aucs]}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(shots, eers, 'o-', color='coral', lw=2, markersize=10)
        ax1.set_xlabel('Enrollment Shots (k)', fontsize=12)
        ax1.set_ylabel('Equal Error Rate (EER)', fontsize=12)
        ax1.set_title('EER vs Enrollment Shots', fontsize=14, fontweight='bold')
        ax1.set_xticks(shots)
        ax1.grid(True, alpha=0.3)
        
        # Add encoder-only baseline
        encoder_only_eer = all_results["user_disjoint"].get("Full (21 ep) (centroid)", {}).get("eer", 0)
        ax1.axhline(encoder_only_eer, color='blue', linestyle='--', label=f'Encoder-only: {encoder_only_eer:.4f}')
        ax1.legend()
        
        ax2.plot(shots, aucs, 'o-', color='steelblue', lw=2, markersize=10)
        ax2.set_xlabel('Enrollment Shots (k)', fontsize=12)
        ax2.set_ylabel('AUC', fontsize=12)
        ax2.set_title('AUC vs Enrollment Shots', fontsize=14, fontweight='bold')
        ax2.set_xticks(shots)
        ax2.grid(True, alpha=0.3)
        
        encoder_only_auc = all_results["user_disjoint"].get("Full (21 ep) (centroid)", {}).get("auc", 0)
        ax2.axhline(encoder_only_auc, color='blue', linestyle='--', label=f'Encoder-only: {encoder_only_auc:.4f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig("results/full/eer_vs_shots.png", dpi=150)
        plt.close()
        
        print(f"\nSaved plot to: results/full/eer_vs_shots.png")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
