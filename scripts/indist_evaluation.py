#!/usr/bin/env python
"""
In-Distribution Evaluation Script
==================================
Corrected evaluation protocol for adapters:
- Train encoder on temporal split (all users, session-based)
- Train adapters on in-distribution users (enrollment samples)
- Evaluate on held-out samples of SAME enrolled users

This answers: Do adapters help once a user is enrolled?
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from contrastive_encoder.data.preprocessing import prepare_sequences, create_splits
from contrastive_encoder.models.encoder import KeystrokeEncoder
from contrastive_encoder.models.adapter import UserAdapter, UserTemplate
from contrastive_encoder.models.losses import TripletLoss
from contrastive_encoder.evaluation.evaluate import compute_eer
from contrastive_encoder.training.train_encoder import EncoderTrainer
from contrastive_encoder.config import EncoderConfig, AdapterConfig
from contrastive_encoder.utils.helpers import set_seed, get_device, save_json


class NoResidualAdapter(nn.Module):
    """Adapter without residual connection for ablation."""
    
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=128, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x, normalize=True):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        # NO residual connection
        if normalize:
            out = F.normalize(out, p=2, dim=1)
        return out


def train_fixed_epoch_encoder(epochs, output_path, checkpoint_dir, log_dir):
    """Train encoder with fixed epochs (no early stopping)."""
    print(f"\n{'='*60}")
    print(f"TRAINING ENCODER ({epochs} epochs, no early stopping)")
    print(f"{'='*60}")
    
    # Load data
    data = prepare_sequences("cmu_keystroke.csv", return_dataframe=True)
    splits = create_splits(data, split_type="temporal", random_seed=42)
    
    # Config with early stopping disabled
    config = EncoderConfig(
        epochs=epochs,
        batch_size=128,
        disable_early_stop=True,
        checkpoint_every=10,
    )
    
    trainer = EncoderTrainer(
        config=config,
        use_wandb=False,
        use_tensorboard=True,
        log_dir=log_dir,
    )
    
    history = trainer.train(
        train_data=splits["train"],
        val_data=splits["val"],
        checkpoint_dir=checkpoint_dir,
        save_path=output_path,
    )
    
    return history


def in_distribution_evaluation(
    encoder_path,
    data,
    k_shots_list=[5, 10],
    use_residual=True,
    device=None,
):
    """
    Evaluate adapters on held-out samples of SAME enrolled users.
    
    Protocol:
    1. Split each user's data into enrollment (k shots) and test (remaining)
    2. Train adapter using encoder embeddings of enrollment samples
    3. Evaluate on test samples of SAME user
    """
    if device is None:
        device = get_device()
    
    # Load encoder
    encoder = KeystrokeEncoder()
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device)
    encoder.eval()
    
    X_seq = data["X_seq_scaled"]
    y = data["y"]
    unique_users = np.unique(y)
    
    results = {}
    
    for k in k_shots_list:
        print(f"\n--- Evaluating k={k} shots ---")
        
        all_genuine_scores = []
        all_impostor_scores = []
        
        for user_label in tqdm(unique_users, desc=f"k={k}"):
            user_mask = y == user_label
            user_indices = np.where(user_mask)[0]
            
            # Need at least k+10 samples (k for enrollment, 10 for testing)
            if len(user_indices) < k + 10:
                continue
            
            # Shuffle and split
            np.random.shuffle(user_indices)
            enrollment_idx = user_indices[:k]
            test_idx = user_indices[k:k+20]  # Up to 20 test samples
            
            # Get negative samples (other users)
            other_mask = ~user_mask
            other_indices = np.where(other_mask)[0]
            np.random.shuffle(other_indices)
            neg_idx = other_indices[:100]
            
            # Compute embeddings
            with torch.no_grad():
                X_enroll = torch.FloatTensor(X_seq[enrollment_idx]).to(device)
                X_test = torch.FloatTensor(X_seq[test_idx]).to(device)
                X_neg = torch.FloatTensor(X_seq[neg_idx]).to(device)
                
                enroll_emb = encoder(X_enroll, normalize=True)
                test_emb = encoder(X_test, normalize=True)
                neg_emb = encoder(X_neg, normalize=True)
            
            # Train adapter
            if use_residual:
                adapter = UserAdapter(hidden_dim=64, dropout=0.2).to(device)
            else:
                adapter = NoResidualAdapter(hidden_dim=64, dropout=0.2).to(device)
            
            optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
            loss_fn = TripletLoss(margin=0.3)
            
            # Train for 100 steps
            adapter.train()
            for step in range(100):
                optimizer.zero_grad()
                
                adapted_enroll = adapter(enroll_emb)
                centroid = adapted_enroll.mean(dim=0, keepdim=True)
                centroid = F.normalize(centroid, p=2, dim=1)
                
                # Find hard negatives
                adapted_neg = adapter(neg_emb)
                neg_sims = F.cosine_similarity(adapted_neg, centroid)
                hard_idx = neg_sims.argsort(descending=True)[:min(32, len(neg_emb))]
                
                # Triplet loss
                anchor = centroid.expand(k, -1)
                positive = adapted_enroll
                neg_sample_idx = torch.randint(0, len(hard_idx), (k,))
                negative = adapted_neg[hard_idx[neg_sample_idx]]
                
                loss = loss_fn(anchor, positive, negative)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            adapter.eval()
            with torch.no_grad():
                adapted_enroll = adapter(enroll_emb)
                centroid = adapted_enroll.mean(dim=0)
                centroid = F.normalize(centroid, dim=0)
                
                # Genuine scores
                adapted_test = adapter(test_emb)
                genuine = F.cosine_similarity(adapted_test, centroid.unsqueeze(0)).cpu().numpy()
                all_genuine_scores.extend(genuine.tolist())
                
                # Impostor scores
                adapted_neg = adapter(neg_emb)
                impostor = F.cosine_similarity(adapted_neg, centroid.unsqueeze(0)).cpu().numpy()
                all_impostor_scores.extend(impostor.tolist())
        
        # Compute metrics
        genuine = np.array(all_genuine_scores)
        impostor = np.array(all_impostor_scores)
        
        y_true = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
        y_scores = np.concatenate([genuine, impostor])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        eer, eer_threshold = compute_eer(y_true, y_scores)
        
        results[k] = {
            "auc": roc_auc,
            "eer": eer,
            "genuine_mean": float(genuine.mean()),
            "impostor_mean": float(impostor.mean()),
            "n_genuine": len(genuine),
            "n_impostor": len(impostor),
        }
        
        print(f"  k={k}: AUC={roc_auc:.4f}, EER={eer:.4f}")
    
    return results


def centroid_only_evaluation(encoder_path, data, device=None):
    """Evaluate encoder with centroid only (no adapter) for comparison."""
    if device is None:
        device = get_device()
    
    encoder = KeystrokeEncoder()
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device)
    encoder.eval()
    
    X_seq = data["X_seq_scaled"]
    y = data["y"]
    unique_users = np.unique(y)
    
    print("\n--- Centroid-only evaluation ---")
    
    all_genuine = []
    all_impostor = []
    
    for user_label in tqdm(unique_users, desc="Centroid"):
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
        neg_idx = other_indices[:100]
        
        with torch.no_grad():
            X_enroll = torch.FloatTensor(X_seq[enroll_idx]).to(device)
            X_test = torch.FloatTensor(X_seq[test_idx]).to(device)
            X_neg = torch.FloatTensor(X_seq[neg_idx]).to(device)
            
            enroll_emb = encoder(X_enroll, normalize=True)
            test_emb = encoder(X_test, normalize=True)
            neg_emb = encoder(X_neg, normalize=True)
            
            centroid = enroll_emb.mean(dim=0)
            centroid = F.normalize(centroid, dim=0)
            
            genuine = F.cosine_similarity(test_emb, centroid.unsqueeze(0)).cpu().numpy()
            impostor = F.cosine_similarity(neg_emb, centroid.unsqueeze(0)).cpu().numpy()
            
            all_genuine.extend(genuine.tolist())
            all_impostor.extend(impostor.tolist())
    
    genuine = np.array(all_genuine)
    impostor = np.array(all_impostor)
    
    y_true = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    y_scores = np.concatenate([genuine, impostor])
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    eer, _ = compute_eer(y_true, y_scores)
    
    print(f"  Centroid-only: AUC={roc_auc:.4f}, EER={eer:.4f}")
    
    return {"auc": roc_auc, "eer": eer}


def main():
    set_seed(42)
    device = get_device()
    
    os.makedirs("results/indist", exist_ok=True)
    
    # Step 1: Train fixed-epoch encoders
    encoders_to_train = [
        (50, "artifacts/encoder_50ep.pt", "artifacts/ckpt_50ep", "logs/encoder_50ep"),
        (100, "artifacts/encoder_100ep.pt", "artifacts/ckpt_100ep", "logs/encoder_100ep"),
    ]
    
    for epochs, path, ckpt_dir, log_dir in encoders_to_train:
        if not os.path.exists(path):
            train_fixed_epoch_encoder(epochs, path, ckpt_dir, log_dir)
        else:
            print(f"Encoder {path} already exists, skipping training")
    
    # Step 2: Load data
    data = prepare_sequences("cmu_keystroke.csv")
    
    # Step 3: Run evaluations
    all_results = {}
    
    for epochs, path, _, _ in encoders_to_train:
        if os.path.exists(path):
            print(f"\n{'='*60}")
            print(f"EVALUATING {epochs}-EPOCH ENCODER")
            print(f"{'='*60}")
            
            # Centroid only
            centroid_result = centroid_only_evaluation(path, data, device)
            all_results[f"Encoder ({epochs}ep) Centroid"] = centroid_result
            
            # Adapter with residual
            adapter_results = in_distribution_evaluation(
                path, data, k_shots_list=[5, 10], use_residual=True, device=device
            )
            for k, res in adapter_results.items():
                all_results[f"Encoder ({epochs}ep) + Adapter (k={k})"] = res
            
            # Ablation: No residual (k=10 only)
            ablation_results = in_distribution_evaluation(
                path, data, k_shots_list=[10], use_residual=False, device=device
            )
            for k, res in ablation_results.items():
                all_results[f"Encoder ({epochs}ep) + NoResidual (k={k})"] = res
    
    # Step 4: Save results
    save_json(all_results, "results/indist/all_results.json")
    
    # Step 5: Create comparison table
    print("\n" + "="*60)
    print("IN-DISTRIBUTION EVALUATION RESULTS")
    print("="*60)
    
    table_data = []
    for model, metrics in all_results.items():
        table_data.append({
            "Model": model,
            "AUC": f"{metrics['auc']:.4f}",
            "EER": f"{metrics['eer']:.4f}",
        })
    
    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))
    df.to_csv("results/indist/comparison_table.csv", index=False)
    
    # Step 6: EER vs shots plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for epochs in [50, 100]:
        shots = []
        eers = []
        for k in [5, 10]:
            key = f"Encoder ({epochs}ep) + Adapter (k={k})"
            if key in all_results:
                shots.append(k)
                eers.append(all_results[key]["eer"])
        
        if shots:
            ax.plot(shots, eers, 'o-', label=f'{epochs} epochs', markersize=8)
    
    # Add centroid baseline
    centroid_key = "Encoder (100ep) Centroid"
    if centroid_key in all_results:
        ax.axhline(all_results[centroid_key]["eer"], color='gray', 
                   linestyle='--', label=f'Centroid-only: {all_results[centroid_key]["eer"]:.4f}')
    
    ax.set_xlabel("Enrollment Shots", fontsize=12)
    ax.set_ylabel("EER", fontsize=12)
    ax.set_title("In-Distribution Evaluation: EER vs Enrollment Shots", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/indist/eer_vs_shots_indist.png", dpi=150)
    plt.close()
    
    print(f"\nSaved results to: results/indist/")


if __name__ == "__main__":
    main()
