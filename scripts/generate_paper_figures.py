#!/usr/bin/env python
"""
Final Paper Figures
===================
Generate publication-ready figures for the paper.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from contrastive_encoder.data.preprocessing import prepare_sequences, create_splits
from contrastive_encoder.models.encoder import KeystrokeEncoder
from contrastive_encoder.evaluation.evaluate import compute_eer
from contrastive_encoder.utils.helpers import set_seed, get_device


def main():
    set_seed(42)
    device = get_device()
    
    os.makedirs("results/paper", exist_ok=True)
    
    # Load data
    data = prepare_sequences("cmu_keystroke.csv", return_dataframe=True)
    
    # Load results
    with open("results/indist/all_results.json", "r") as f:
        indist_results = json.load(f)
    
    # =========================================================================
    # FIGURE 1: Final Comparison Table
    # =========================================================================
    print("Generating final comparison table...")
    
    # Collect results from various sources
    final_results = {
        "Random Forest (baseline)": {"eer": 0.263, "auc": 0.80, "setting": "Cross-user"},
        "XGBoost (baseline)": {"eer": 0.394, "auc": 0.66, "setting": "Cross-user"},
        "Encoder (cross-user)": {"eer": 0.177, "auc": 0.897, "setting": "Cross-user"},
        "Encoder (in-dist, centroid)": {"eer": 0.033, "auc": 0.994, "setting": "In-distribution"},
        "Encoder + Adapter (k=5)": {"eer": 0.028, "auc": 0.993, "setting": "In-distribution"},
        "Encoder + NoResidual (k=10)": {"eer": 0.026, "auc": 0.991, "setting": "In-distribution"},
    }
    
    # Save as CSV
    import pandas as pd
    rows = []
    for model, metrics in final_results.items():
        rows.append({
            "Model": model,
            "Setting": metrics["setting"],
            "AUC": f"{metrics['auc']:.3f}",
            "EER (%)": f"{metrics['eer']*100:.1f}",
        })
    df = pd.DataFrame(rows)
    df.to_csv("results/paper/final_comparison_table.csv", index=False)
    print(df.to_string(index=False))
    
    # =========================================================================
    # FIGURE 2: Consolidated ROC Curves
    # =========================================================================
    print("\nGenerating consolidated ROC figure...")
    
    # Load encoder
    encoder = KeystrokeEncoder()
    encoder.load_state_dict(torch.load("artifacts/encoder_100ep.pt", map_location=device))
    encoder.to(device).eval()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # --- Cross-user ROC ---
    splits = create_splits(data, "user_disjoint", random_seed=42)
    X_train, y_train = splits["train"]["X_seq"], splits["train"]["y"]
    X_test, y_test = splits["test"]["X_seq"], splits["test"]["y"]
    
    # Compute genuine/impostor scores for cross-user
    with torch.no_grad():
        X_train_t = torch.FloatTensor(X_train).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        
        train_emb = encoder(X_train_t, normalize=True).cpu().numpy()
        test_emb = encoder(X_test_t, normalize=True).cpu().numpy()
    
    # Compute user centroids from training data
    unique_train_users = np.unique(y_train)
    centroids = {}
    for u in unique_train_users:
        mask = y_train == u
        centroids[u] = train_emb[mask].mean(axis=0)
        centroids[u] /= np.linalg.norm(centroids[u])
    
    # For cross-user, test users are NOT in training
    unique_test_users = np.unique(y_test)
    genuine_cross, impostor_cross = [], []
    
    for u in unique_test_users:
        user_mask = y_test == u
        user_emb = test_emb[user_mask]
        
        # Use first 5 samples as enrollment, rest as test
        if len(user_emb) < 10:
            continue
        
        centroid = user_emb[:5].mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        
        # Genuine
        for emb in user_emb[5:]:
            sim = np.dot(emb, centroid)
            genuine_cross.append(sim)
        
        # Impostor (samples from other test users)
        for other_u in unique_test_users:
            if other_u == u:
                continue
            other_mask = y_test == other_u
            other_emb = test_emb[other_mask][:10]
            for emb in other_emb:
                sim = np.dot(emb, centroid)
                impostor_cross.append(sim)
    
    y_true_cross = np.concatenate([np.ones(len(genuine_cross)), np.zeros(len(impostor_cross))])
    y_scores_cross = np.concatenate([genuine_cross, impostor_cross])
    fpr_cross, tpr_cross, _ = roc_curve(y_true_cross, y_scores_cross)
    auc_cross = auc(fpr_cross, tpr_cross)
    eer_cross, _ = compute_eer(y_true_cross, y_scores_cross)
    
    ax.plot(fpr_cross, tpr_cross, 'b-', lw=2, 
            label=f'Encoder (cross-user) — AUC={auc_cross:.3f}, EER={eer_cross*100:.1f}%')
    
    # --- In-distribution ROC ---
    # Re-use full data
    X_seq = data["X_seq_scaled"]
    y = data["y"]
    unique_users = np.unique(y)
    
    genuine_indist, impostor_indist = [], []
    
    for user_label in unique_users:
        user_mask = y == user_label
        user_indices = np.where(user_mask)[0]
        
        if len(user_indices) < 15:
            continue
        
        np.random.shuffle(user_indices)
        enroll_idx = user_indices[:5]
        test_idx = user_indices[5:20]
        
        other_mask = ~user_mask
        other_indices = np.where(other_mask)[0]
        np.random.shuffle(other_indices)
        neg_idx = other_indices[:50]
        
        with torch.no_grad():
            X_enroll = torch.FloatTensor(X_seq[enroll_idx]).to(device)
            X_test_user = torch.FloatTensor(X_seq[test_idx]).to(device)
            X_neg = torch.FloatTensor(X_seq[neg_idx]).to(device)
            
            enroll_emb = encoder(X_enroll, normalize=True)
            test_emb_user = encoder(X_test_user, normalize=True)
            neg_emb = encoder(X_neg, normalize=True)
            
            centroid = enroll_emb.mean(dim=0)
            centroid = F.normalize(centroid, dim=0)
            
            genuine = F.cosine_similarity(test_emb_user, centroid.unsqueeze(0)).cpu().numpy()
            impostor = F.cosine_similarity(neg_emb, centroid.unsqueeze(0)).cpu().numpy()
            
            genuine_indist.extend(genuine.tolist())
            impostor_indist.extend(impostor.tolist())
    
    y_true_indist = np.concatenate([np.ones(len(genuine_indist)), np.zeros(len(impostor_indist))])
    y_scores_indist = np.concatenate([genuine_indist, impostor_indist])
    fpr_indist, tpr_indist, _ = roc_curve(y_true_indist, y_scores_indist)
    auc_indist = auc(fpr_indist, tpr_indist)
    eer_indist, _ = compute_eer(y_true_indist, y_scores_indist)
    
    ax.plot(fpr_indist, tpr_indist, 'g-', lw=2,
            label=f'Encoder (in-distribution) — AUC={auc_indist:.3f}, EER={eer_indist*100:.1f}%')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves: Cross-User vs In-Distribution Evaluation', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig("results/paper/roc_consolidated.png", dpi=300)
    plt.close()
    print("Saved: results/paper/roc_consolidated.png")
    
    # =========================================================================
    # FIGURE 3: EER vs Enrollment Shots
    # =========================================================================
    print("\nGenerating EER vs enrollment shots figure...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract from indist results
    shots = [5, 10]
    adapter_eers = [
        indist_results.get("Encoder (100ep) + Adapter (k=5)", {}).get("eer", 0),
        indist_results.get("Encoder (100ep) + Adapter (k=10)", {}).get("eer", 0),
    ]
    noresidual_eers = [
        None,  # No k=5 for noresidual
        indist_results.get("Encoder (100ep) + NoResidual (k=10)", {}).get("eer", 0),
    ]
    
    # Plot
    ax.plot(shots, adapter_eers, 'o-', color='steelblue', lw=2, markersize=10,
            label='Encoder + Adapter (residual)')
    
    # Add NoResidual point
    ax.scatter([10], [noresidual_eers[1]], color='coral', s=150, marker='s', zorder=5,
               label=f'Encoder + NoResidual (k=10): {noresidual_eers[1]*100:.1f}%')
    
    # Centroid baseline
    centroid_eer = indist_results.get("Encoder (100ep) Centroid", {}).get("eer", 0)
    ax.axhline(centroid_eer, color='gray', linestyle='--', lw=2, 
               label=f'Centroid-only: {centroid_eer*100:.1f}%')
    
    ax.set_xlabel('Enrollment Shots (k)', fontsize=12)
    ax.set_ylabel('Equal Error Rate (EER)', fontsize=12)
    ax.set_title('EER vs Enrollment Shots (In-Distribution)', fontsize=14, fontweight='bold')
    ax.set_xticks(shots)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
    
    plt.tight_layout()
    plt.savefig("results/paper/eer_vs_shots.png", dpi=300)
    plt.close()
    print("Saved: results/paper/eer_vs_shots.png")
    
    print("\n" + "="*60)
    print("ALL PAPER FIGURES GENERATED")
    print("="*60)
    print("  - results/paper/final_comparison_table.csv")
    print("  - results/paper/roc_consolidated.png")
    print("  - results/paper/eer_vs_shots.png")


if __name__ == "__main__":
    main()
