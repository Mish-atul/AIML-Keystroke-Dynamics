"""
Visualization Module
====================
Plotting functions for evaluation results.
"""

import os
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_roc_curves(
    results: Dict[str, Dict[str, Any]],
    output_path: str,
    title: str = "ROC Curves - Verification",
) -> None:
    """
    Plot ROC curves for multiple models.
    
    Args:
        results: Dictionary mapping model names to result dicts with 'fpr' and 'tpr'
        output_path: Path to save plot
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for (model_name, metrics), color in zip(results.items(), colors):
        if 'fpr' in metrics and 'tpr' in metrics:
            fpr = np.array(metrics['fpr'])
            tpr = np.array(metrics['tpr'])
            auc_val = metrics.get('auc', 0)
            eer_val = metrics.get('eer', 0)
            
            plt.plot(
                fpr, tpr,
                color=color,
                lw=2,
                label=f"{model_name} (AUC={auc_val:.4f}, EER={eer_val:.4f})"
            )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Acceptance Rate (FAR)', fontsize=12)
    plt.ylabel('True Positive Rate (1 - FRR)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ROC plot to: {output_path}")


def plot_eer_vs_shots(
    shot_results: Dict[int, Dict[str, float]],
    output_path: str,
) -> None:
    """
    Plot EER vs enrollment shots.
    
    Args:
        shot_results: Dictionary mapping shot count to metrics dict
        output_path: Path to save plot
    """
    shots = sorted(shot_results.keys())
    eers = [shot_results[s]['eer'] for s in shots]
    aucs = [shot_results[s]['auc'] for s in shots]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # EER plot
    ax1.plot(shots, eers, 'o-', color='coral', lw=2, markersize=8)
    ax1.set_xlabel('Enrollment Shots', fontsize=12)
    ax1.set_ylabel('Equal Error Rate (EER)', fontsize=12)
    ax1.set_title('EER vs Enrollment Shots', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(shots)
    
    # AUC plot
    ax2.plot(shots, aucs, 'o-', color='steelblue', lw=2, markersize=8)
    ax2.set_xlabel('Enrollment Shots', fontsize=12)
    ax2.set_ylabel('AUC', fontsize=12)
    ax2.set_title('AUC vs Enrollment Shots', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(shots)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved shots plot to: {output_path}")


def plot_score_distributions(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: float,
    output_path: str,
) -> None:
    """
    Plot genuine vs impostor score distributions.
    
    Args:
        genuine_scores: Similarity scores for genuine pairs
        impostor_scores: Similarity scores for impostor pairs
        threshold: Decision threshold
        output_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(
        genuine_scores, bins=50, alpha=0.6,
        label=f'Genuine (n={len(genuine_scores)})',
        color='green', density=True
    )
    plt.hist(
        impostor_scores, bins=50, alpha=0.6,
        label=f'Impostor (n={len(impostor_scores)})',
        color='red', density=True
    )
    
    plt.axvline(threshold, color='black', linestyle='--', lw=2, label=f'Threshold ({threshold:.3f})')
    
    plt.xlabel('Similarity Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Score Distributions: Genuine vs Impostor', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved distribution plot to: {output_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    max_classes: int = 20,
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        output_path: Path to save plot
        labels: Optional class labels
        title: Plot title
        max_classes: Maximum classes to display
    """
    from sklearn.metrics import confusion_matrix as compute_cm
    
    cm = compute_cm(y_true, y_pred)
    
    # Limit size for visualization
    if len(cm) > max_classes:
        cm = cm[:max_classes, :max_classes]
        if labels:
            labels = labels[:max_classes]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion matrix to: {output_path}")


def plot_embeddings_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    n_samples: int = 2000,
    perplexity: int = 30,
    title: str = "Embedding Visualization (t-SNE)",
) -> None:
    """
    Plot t-SNE visualization of embeddings.
    
    Args:
        embeddings: Embedding vectors (N, D)
        labels: Labels for coloring
        output_path: Path to save plot
        n_samples: Max samples to plot
        perplexity: t-SNE perplexity
        title: Plot title
    """
    from sklearn.manifold import TSNE
    
    # Subsample if needed
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    
    print(f"Computing t-SNE for {len(embeddings)} samples...")
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique_labels))))
    
    for i, label in enumerate(unique_labels[:20]):  # Limit legend
        mask = labels == label
        plt.scatter(
            coords[mask, 0], coords[mask, 1],
            s=10, alpha=0.6, c=[colors[i % 20]],
            label=f'User {label}'
        )
    
    if len(unique_labels) > 20:
        for label in unique_labels[20:]:
            mask = labels == label
            plt.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.3, c='gray')
    
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    if len(unique_labels) <= 20:
        plt.legend(fontsize=8, markerscale=2, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved t-SNE plot to: {output_path}")


def plot_embeddings_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    n_samples: int = 2000,
    n_neighbors: int = 15,
    title: str = "Embedding Visualization (UMAP)",
) -> None:
    """
    Plot UMAP visualization of embeddings.
    
    Args:
        embeddings: Embedding vectors
        labels: Labels for coloring
        output_path: Path to save plot
        n_samples: Max samples
        n_neighbors: UMAP neighbors
        title: Plot title
    """
    try:
        import umap
    except ImportError:
        print("UMAP not available, skipping visualization")
        return
    
    # Subsample if needed
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    
    print(f"Computing UMAP for {len(embeddings)} samples...")
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42)
    coords = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique_labels))))
    
    for i, label in enumerate(unique_labels[:20]):
        mask = labels == label
        plt.scatter(
            coords[mask, 0], coords[mask, 1],
            s=10, alpha=0.6, c=[colors[i % 20]],
            label=f'User {label}'
        )
    
    if len(unique_labels) > 20:
        for label in unique_labels[20:]:
            mask = labels == label
            plt.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.3, c='gray')
    
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    if len(unique_labels) <= 20:
        plt.legend(fontsize=8, markerscale=2, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved UMAP plot to: {output_path}")


def plot_per_user_eer(
    per_user_results: Dict[str, Dict[str, float]],
    output_path: str,
    top_n: int = 20,
) -> None:
    """
    Plot per-user EER bar chart.
    
    Args:
        per_user_results: Dictionary mapping user_id to metrics
        output_path: Path to save plot
        top_n: Number of best/worst users to highlight
    """
    users = list(per_user_results.keys())
    eers = [per_user_results[u]['eer'] for u in users]
    
    # Sort by EER
    sorted_pairs = sorted(zip(users, eers), key=lambda x: x[1])
    sorted_users, sorted_eers = zip(*sorted_pairs)
    
    plt.figure(figsize=(14, 6))
    
    colors = ['green' if e < 0.1 else 'orange' if e < 0.2 else 'red' for e in sorted_eers]
    
    plt.bar(range(len(sorted_users)), sorted_eers, color=colors)
    
    plt.xlabel('User (sorted by EER)', fontsize=12)
    plt.ylabel('Equal Error Rate', fontsize=12)
    plt.title('Per-User EER Distribution', fontsize=14, fontweight='bold')
    
    # Add mean line
    mean_eer = np.mean(sorted_eers)
    plt.axhline(mean_eer, color='black', linestyle='--', lw=2, label=f'Mean EER: {mean_eer:.4f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved per-user EER plot to: {output_path}")
