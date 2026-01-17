"""
Evaluation Module
=================
Metrics computation and baseline comparisons.
"""

import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
)
from tqdm import tqdm
import pickle

from ..models.encoder import KeystrokeEncoder
from ..models.adapter import UserAdapter, UserTemplate
from ..utils.helpers import save_json, load_json, get_device


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER).
    
    Args:
        y_true: Binary labels (1 = genuine, 0 = impostor)
        y_scores: Similarity/confidence scores
        
    Returns:
        Tuple of (EER, threshold at EER)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # Find the threshold where FPR ≈ FNR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx] if eer_idx < len(thresholds) else 0.5
    
    return float(eer), float(eer_threshold)


def evaluate_encoder_identification(
    encoder: KeystrokeEncoder,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluate encoder for identification (nearest neighbor classification).
    
    Args:
        encoder: Trained encoder
        X_test: Test sequences (N_test, 11, 3)
        y_test: Test labels
        X_train: Training sequences (N_train, 11, 3) as gallery
        y_train: Training labels
        device: Compute device
        
    Returns:
        Metrics dictionary
    """
    encoder.eval()
    
    with torch.no_grad():
        # Compute embeddings
        X_train_t = torch.FloatTensor(X_train).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        
        train_emb = encoder(X_train_t, normalize=True)
        test_emb = encoder(X_test_t, normalize=True)
        
        # Nearest neighbor
        sim = torch.mm(test_emb, train_emb.T)  # (N_test, N_train)
        
        # Top-1
        top1_idx = sim.argmax(dim=1).cpu().numpy()
        top1_preds = y_train[top1_idx]
        top1_acc = accuracy_score(y_test, top1_preds)
        
        # Top-5
        top5_idx = sim.topk(5, dim=1).indices.cpu().numpy()
        top5_correct = 0
        for i, gt in enumerate(y_test):
            if gt in y_train[top5_idx[i]]:
                top5_correct += 1
        top5_acc = top5_correct / len(y_test)
    
    return {
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "n_test_samples": len(y_test),
        "n_train_samples": len(y_train),
    }


def evaluate_encoder_verification(
    encoder: KeystrokeEncoder,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    n_pairs: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate encoder for verification (pairwise comparison).
    
    Args:
        encoder: Trained encoder
        X: Sequences (N, 11, 3)
        y: Labels
        device: Compute device
        n_pairs: Number of pairs to sample
        seed: Random seed
        
    Returns:
        Metrics dictionary
    """
    np.random.seed(seed)
    encoder.eval()
    
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        embeddings = encoder(X_t, normalize=True).cpu().numpy()
    
    # Sample genuine pairs (same user)
    genuine_pairs = []
    impostor_pairs = []
    
    unique_labels = np.unique(y)
    label_indices = {l: np.where(y == l)[0] for l in unique_labels}
    
    # Sample pairs
    for _ in range(n_pairs // 2):
        # Genuine pair
        label = np.random.choice(unique_labels)
        indices = label_indices[label]
        if len(indices) >= 2:
            i, j = np.random.choice(indices, 2, replace=False)
            sim = np.dot(embeddings[i], embeddings[j])
            genuine_pairs.append(sim)
        
        # Impostor pair
        l1, l2 = np.random.choice(unique_labels, 2, replace=False)
        i = np.random.choice(label_indices[l1])
        j = np.random.choice(label_indices[l2])
        sim = np.dot(embeddings[i], embeddings[j])
        impostor_pairs.append(sim)
    
    genuine_scores = np.array(genuine_pairs)
    impostor_scores = np.array(impostor_pairs)
    
    # Combined
    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    y_scores = np.concatenate([genuine_scores, impostor_scores])
    
    # Metrics
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    eer, eer_threshold = compute_eer(y_true, y_scores)
    
    return {
        "auc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "genuine_mean": float(genuine_scores.mean()),
        "genuine_std": float(genuine_scores.std()),
        "impostor_mean": float(impostor_scores.mean()),
        "impostor_std": float(impostor_scores.std()),
        "n_genuine_pairs": len(genuine_scores),
        "n_impostor_pairs": len(impostor_scores),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def evaluate_with_adapters(
    encoder: KeystrokeEncoder,
    templates: Dict[str, UserTemplate],
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluate encoder + adapters for verification.
    
    Args:
        encoder: Trained encoder
        templates: Dictionary of user templates
        X_test: Test sequences
        y_test: Test labels (integers)
        label_encoder: LabelEncoder to map back to user IDs
        device: Compute device
        
    Returns:
        Metrics dictionary
    """
    encoder.eval()
    
    all_genuine_scores = []
    all_impostor_scores = []
    per_user_results = {}
    
    with torch.no_grad():
        for user_id, template in tqdm(templates.items(), desc="Evaluating users"):
            # Get user's test samples
            try:
                user_label = label_encoder.transform([user_id])[0]
            except:
                # Try numeric extraction
                user_label = int(user_id.replace("s", ""))
            
            user_mask = y_test == user_label
            other_mask = ~user_mask
            
            if user_mask.sum() == 0:
                continue
            
            # Load adapter
            adapter = template.adapter.to(device)
            adapter.eval()
            
            centroid = template.centroid.to(device)
            threshold = template.threshold
            
            # Compute scores for genuine samples
            X_user = torch.FloatTensor(X_test[user_mask]).to(device)
            emb = encoder(X_user, normalize=True)
            adapted = adapter(emb, normalize=True)
            genuine_scores = F.cosine_similarity(
                adapted, centroid.unsqueeze(0).expand(len(adapted), -1)
            ).cpu().numpy()
            
            all_genuine_scores.extend(genuine_scores.tolist())
            
            # Compute scores for impostor samples (sample subset)
            n_impostors = min(user_mask.sum() * 10, other_mask.sum())
            impostor_indices = np.random.choice(
                np.where(other_mask)[0], size=n_impostors, replace=False
            )
            
            X_impostor = torch.FloatTensor(X_test[impostor_indices]).to(device)
            emb = encoder(X_impostor, normalize=True)
            adapted = adapter(emb, normalize=True)
            impostor_scores = F.cosine_similarity(
                adapted, centroid.unsqueeze(0).expand(len(adapted), -1)
            ).cpu().numpy()
            
            all_impostor_scores.extend(impostor_scores.tolist())
            
            # Per-user EER
            user_y_true = np.concatenate([
                np.ones(len(genuine_scores)),
                np.zeros(len(impostor_scores))
            ])
            user_y_scores = np.concatenate([genuine_scores, impostor_scores])
            user_eer, _ = compute_eer(user_y_true, user_y_scores)
            
            per_user_results[user_id] = {
                "eer": user_eer,
                "n_genuine": len(genuine_scores),
                "n_impostor": len(impostor_scores),
                "genuine_mean": float(genuine_scores.mean()),
                "impostor_mean": float(impostor_scores.mean()),
            }
    
    # Global metrics
    genuine_scores = np.array(all_genuine_scores)
    impostor_scores = np.array(all_impostor_scores)
    
    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    y_scores = np.concatenate([genuine_scores, impostor_scores])
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    eer, eer_threshold = compute_eer(y_true, y_scores)
    
    return {
        "auc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "genuine_mean": float(genuine_scores.mean()),
        "impostor_mean": float(impostor_scores.mean()),
        "per_user": per_user_results,
        "n_users_evaluated": len(per_user_results),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def load_baseline_model(model_type: str, model_path: str):
    """Load a baseline model."""
    if model_type == "rf":
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    elif model_type == "xgb":
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model
    elif model_type == "hgb":
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    elif model_type in ["mlp", "cnn"]:
        from tensorflow import keras
        return keras.models.load_model(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_baseline(
    model,
    model_type: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler=None,
) -> Dict[str, Any]:
    """
    Evaluate a baseline model.
    
    Args:
        model: Loaded model
        model_type: Type string (rf, xgb, hgb, mlp, cnn)
        X_test: Test features
        y_test: Test labels
        scaler: Optional scaler to apply
        
    Returns:
        Metrics dictionary
    """
    if scaler is not None:
        X_test = scaler.transform(X_test)
    
    # Get predictions
    if model_type in ["mlp", "cnn"]:
        pred_proba = model.predict(X_test, verbose=0)
        predictions = np.argmax(pred_proba, axis=1)
    else:
        predictions = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)
    
    # Identification accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    # Verification metrics
    max_proba = np.max(pred_proba, axis=1)
    is_correct = (predictions == y_test).astype(int)
    
    fpr, tpr, _ = roc_curve(is_correct, max_proba)
    roc_auc = auc(fpr, tpr)
    eer, _ = compute_eer(is_correct, max_proba)
    
    return {
        "accuracy": accuracy,
        "auc": roc_auc,
        "eer": eer,
        "n_samples": len(y_test),
    }


def build_comparison_table(
    results: Dict[str, Dict[str, Any]],
    output_path: str,
) -> pd.DataFrame:
    """
    Build and save comparison table.
    
    Args:
        results: Dictionary mapping model names to result dicts
        output_path: Path to save CSV
        
    Returns:
        DataFrame with comparison
    """
    rows = []
    
    for model_name, metrics in results.items():
        row = {
            "Model": model_name,
            "Accuracy": metrics.get("accuracy", metrics.get("top1_accuracy", "-")),
            "Top-5 Acc": metrics.get("top5_accuracy", "-"),
            "AUC": metrics.get("auc", "-"),
            "EER": metrics.get("eer", "-"),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Format percentages
    for col in ["Accuracy", "Top-5 Acc", "EER"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: f"{x:.4f}" if isinstance(x, float) else x
            )
    
    df.to_csv(output_path, index=False)
    print(f"Saved comparison table to: {output_path}")
    
    return df
