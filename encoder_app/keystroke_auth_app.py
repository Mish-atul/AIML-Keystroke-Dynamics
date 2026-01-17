"""
Keystroke Dynamics Authentication App - Full Research-Grade Implementation
===========================================================================
Features:
- Contrastive encoder with proper adapter training (triplet loss + real CMU negatives)
- Backend visualizations (H/DD/UD timing, enrollment vs login comparison)
- Continuous learning with high-confidence EMA merge
- No-residual linear adapter (few-shot safe)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import sys
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pickle
from collections import deque

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# =============================================================================
# Configuration
# =============================================================================
PASSWORD = ".tie5Roanl"
KEY_ORDER = ["period", "t", "i", "e", "five", "Shift.r", "o", "a", "n", "l", "Return"]
FEATURE_NAMES = ["H (Hold)", "DD (Down-Down)", "UD (Up-Down)"]

# Paths
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
ENCODER_PATH = ARTIFACTS_DIR / "encoder_100ep.pt"
SCALER_PATH = ARTIFACTS_DIR / "scaler_seq.pkl"
CMU_DATA_PATH = Path(__file__).parent.parent / "cmu_keystroke.csv"
USERS_DIR = Path(__file__).parent / "users"

# Training config
ADAPTER_HIDDEN_DIM = 64
TRIPLET_MARGIN = 0.3
ADAPTER_LR = 0.01
ADAPTER_EPOCHS = 50
NUM_NEGATIVES_PER_POSITIVE = 5

# Continuous learning config
EMA_MOMENTUM = 0.1
MERGE_MARGIN = 0.08  # Only merge if similarity > threshold + margin
MIN_SAMPLES_FOR_MERGE = 3

# Authentication config
DEFAULT_THRESHOLD = 0.80
MIN_ENROLLMENT_SAMPLES = 5

# =============================================================================
# Neural Network Models (matching training architecture exactly)
# =============================================================================

class Conv1DBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, use_batch_norm=True, pool_size=None):
        super().__init__()
        
        layers = [nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if pool_size is not None:
            layers.append(nn.MaxPool1d(pool_size))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class KeystrokeEncoder(nn.Module):
    """
    1D-CNN Encoder for keystroke sequences - EXACT match to training architecture.
    
    Input shape: (batch, seq_len, features) = (batch, 11, 3)
    Output shape: (batch, embedding_dim) = (batch, 128)
    """
    
    def __init__(self, input_features=3, seq_length=11, embedding_dim=128,
                 conv_channels=None, kernel_size=3, use_batch_norm=True):
        super().__init__()
        
        if conv_channels is None:
            conv_channels = [64, 128, 256]
        
        self.input_features = input_features
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        
        # Build convolutional layers
        conv_layers = []
        in_ch = input_features
        
        for i, out_ch in enumerate(conv_channels):
            pool = 2 if i < 2 else None
            conv_layers.append(
                Conv1DBlock(
                    in_ch, out_ch,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    use_batch_norm=use_batch_norm,
                    pool_size=pool,
                )
            )
            in_ch = out_ch
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(conv_channels[-1], embedding_dim),
            nn.BatchNorm1d(embedding_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x, normalize=True):
        # Transpose: (batch, seq, feat) -> (batch, feat, seq)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.projection(x)
        
        if normalize:
            x = F.normalize(x, p=2, dim=1)
        return x


class LinearAdapter(nn.Module):
    """
    No-residual linear adapter for user-specific fine-tuning.
    Low capacity to prevent overfitting with few-shot enrollment.
    """
    
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=-1)  # L2 normalize output
        return x


class TripletLoss(nn.Module):
    """Triplet loss with semi-hard negative mining using cosine distance."""
    
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        # Cosine distance = 1 - cosine_similarity
        pos_dist = 1 - F.cosine_similarity(anchor, positive)
        neg_dist = 1 - F.cosine_similarity(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


# =============================================================================
# CMU Data Loader
# =============================================================================

class CMUDataLoader:
    """Load and cache CMU keystroke data for negative sampling."""
    
    def __init__(self, csv_path, scaler=None):
        self.csv_path = Path(csv_path)
        self.scaler = scaler
        self.data_by_user = {}
        self.all_users = []
        self._load_data()
        
    def _load_data(self):
        """Parse CMU CSV and organize by user."""
        if not self.csv_path.exists():
            print(f"Warning: CMU data not found at {self.csv_path}")
            return
            
        import csv
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Group by subject
        for row in rows:
            subject = row['subject']
            if subject not in self.data_by_user:
                self.data_by_user[subject] = []
                self.all_users.append(subject)
            
            # Extract features: H, DD, UD for each key
            sequence = []
            for key in KEY_ORDER:
                h_col = f"H.{key}"
                
                # Find DD and UD columns
                key_idx = KEY_ORDER.index(key)
                if key_idx == 0:
                    dd_col = None
                    ud_col = None
                else:
                    prev_key = KEY_ORDER[key_idx - 1]
                    dd_col = f"DD.{prev_key}.{key}"
                    ud_col = f"UD.{prev_key}.{key}"
                
                h_val = float(row.get(h_col, 0))
                dd_val = float(row.get(dd_col, 0)) if dd_col else 0.0
                ud_val = float(row.get(ud_col, 0)) if ud_col else 0.0
                
                sequence.append([h_val, dd_val, ud_val])
            
            self.data_by_user[subject].append(np.array(sequence, dtype=np.float32))
        
        print(f"Loaded CMU data: {len(self.all_users)} users, {sum(len(v) for v in self.data_by_user.values())} samples")
    
    def get_negative_samples(self, exclude_user=None, n_samples=10):
        """Get random negative samples from other users."""
        available_users = [u for u in self.all_users if u != exclude_user]
        if not available_users:
            return []
        
        negatives = []
        for _ in range(n_samples):
            user = np.random.choice(available_users)
            samples = self.data_by_user[user]
            sample = samples[np.random.randint(len(samples))]
            negatives.append(sample)
        
        return negatives
    
    def get_user_samples(self, user_id, n_samples=None):
        """Get samples for a specific user."""
        if user_id not in self.data_by_user:
            return []
        samples = self.data_by_user[user_id]
        if n_samples:
            indices = np.random.choice(len(samples), min(n_samples, len(samples)), replace=False)
            return [samples[i] for i in indices]
        return samples


# =============================================================================
# User Profile Manager
# =============================================================================

class UserProfile:
    """Manages user enrollment data, adapter, and continuous learning state."""
    
    def __init__(self, username, users_dir=USERS_DIR):
        self.username = username
        self.users_dir = Path(users_dir)
        self.users_dir.mkdir(parents=True, exist_ok=True)
        
        self.profile_path = self.users_dir / f"{username}.json"
        self.adapter_path = self.users_dir / f"{username}_adapter.pt"
        
        # Profile data
        self.enrollment_sequences = []  # Raw keystroke sequences
        self.enrollment_embeddings = []  # Encoder embeddings (before adapter)
        self.centroid = None  # Current centroid (after adapter if available)
        self.raw_centroid = None  # Centroid without adapter
        self.threshold = DEFAULT_THRESHOLD
        self.adapter = None
        self.adapter_trained = False
        
        # Continuous learning state
        self.successful_logins = []  # Recent high-confidence logins for EMA
        self.login_count = 0
        self.merge_count = 0
        
        # Statistics for visualization
        self.timing_stats = {
            'H': {'mean': [], 'std': []},
            'DD': {'mean': [], 'std': []},
            'UD': {'mean': [], 'std': []}
        }
        
    def save(self):
        """Save profile to disk."""
        data = {
            'username': self.username,
            'enrollment_sequences': [s.tolist() for s in self.enrollment_sequences],
            'enrollment_embeddings': [e.tolist() for e in self.enrollment_embeddings],
            'centroid': self.centroid.tolist() if self.centroid is not None else None,
            'raw_centroid': self.raw_centroid.tolist() if self.raw_centroid is not None else None,
            'threshold': self.threshold,
            'adapter_trained': self.adapter_trained,
            'login_count': self.login_count,
            'merge_count': self.merge_count,
            'timing_stats': self.timing_stats
        }
        
        with open(self.profile_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        if self.adapter is not None:
            torch.save(self.adapter.state_dict(), self.adapter_path)
    
    def load(self):
        """Load profile from disk."""
        if not self.profile_path.exists():
            return False
        
        with open(self.profile_path, 'r') as f:
            data = json.load(f)
        
        self.enrollment_sequences = [np.array(s, dtype=np.float32) for s in data['enrollment_sequences']]
        self.enrollment_embeddings = [np.array(e, dtype=np.float32) for e in data['enrollment_embeddings']]
        self.centroid = np.array(data['centroid'], dtype=np.float32) if data['centroid'] else None
        self.raw_centroid = np.array(data['raw_centroid'], dtype=np.float32) if data.get('raw_centroid') else None
        self.threshold = data.get('threshold', DEFAULT_THRESHOLD)
        self.adapter_trained = data.get('adapter_trained', False)
        self.login_count = data.get('login_count', 0)
        self.merge_count = data.get('merge_count', 0)
        self.timing_stats = data.get('timing_stats', self.timing_stats)
        
        if self.adapter_path.exists():
            self.adapter = LinearAdapter()
            self.adapter.load_state_dict(torch.load(self.adapter_path, map_location='cpu'))
            self.adapter.eval()
        
        return True
    
    def compute_timing_stats(self):
        """Compute timing statistics from enrollment sequences."""
        if not self.enrollment_sequences:
            return
        
        sequences = np.stack(self.enrollment_sequences)  # (N, 11, 3)
        
        # H, DD, UD are indices 0, 1, 2
        for i, name in enumerate(['H', 'DD', 'UD']):
            values = sequences[:, :, i]  # (N, 11)
            self.timing_stats[name]['mean'] = values.mean(axis=0).tolist()
            self.timing_stats[name]['std'] = values.std(axis=0).tolist()


# =============================================================================
# Adapter Trainer
# =============================================================================

class AdapterTrainer:
    """Train user-specific adapter with triplet loss and real CMU negatives."""
    
    def __init__(self, encoder, cmu_loader, device='cpu'):
        self.encoder = encoder
        self.cmu_loader = cmu_loader
        self.device = device
        
    def train(self, user_profile, progress_callback=None):
        """Train adapter for a user profile."""
        if len(user_profile.enrollment_sequences) < 2:
            print("Need at least 2 enrollment samples for adapter training")
            return False
        
        # Get positive samples (user's enrollment)
        positives = user_profile.enrollment_sequences
        
        # Get negative samples from CMU dataset
        negatives = self.cmu_loader.get_negative_samples(
            exclude_user=None,  # We don't know CMU user ID, exclude none
            n_samples=len(positives) * NUM_NEGATIVES_PER_POSITIVE
        )
        
        if len(negatives) < len(positives):
            print("Not enough negative samples for training")
            return False
        
        # Create adapter
        adapter = LinearAdapter().to(self.device)
        optimizer = torch.optim.Adam(adapter.parameters(), lr=ADAPTER_LR)
        triplet_loss = TripletLoss(margin=TRIPLET_MARGIN)
        
        # Convert to tensors and get embeddings
        pos_seqs = torch.tensor(np.stack(positives), dtype=torch.float32).to(self.device)
        neg_seqs = torch.tensor(np.stack(negatives), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            pos_embeds = self.encoder(pos_seqs)
            neg_embeds = self.encoder(neg_seqs)
        
        # Training loop
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(ADAPTER_EPOCHS):
            adapter.train()
            total_loss = 0
            n_triplets = 0
            
            # For each positive, sample anchor-positive pairs and semi-hard negatives
            for i in range(len(positives)):
                anchor_embed = pos_embeds[i:i+1]
                
                # Sample another positive as the positive pair
                pos_indices = [j for j in range(len(positives)) if j != i]
                if not pos_indices:
                    continue
                pos_idx = np.random.choice(pos_indices)
                positive_embed = pos_embeds[pos_idx:pos_idx+1]
                
                # Apply adapter
                anchor_adapted = adapter(anchor_embed)
                positive_adapted = adapter(positive_embed)
                
                # Semi-hard negative mining
                with torch.no_grad():
                    neg_adapted = adapter(neg_embeds)
                    anchor_pos_dist = 1 - F.cosine_similarity(anchor_adapted, positive_adapted)
                    anchor_neg_dists = 1 - F.cosine_similarity(anchor_adapted.expand_as(neg_adapted), neg_adapted)
                    
                    # Semi-hard: negatives that are farther than positive but within margin
                    semi_hard_mask = (anchor_neg_dists > anchor_pos_dist) & (anchor_neg_dists < anchor_pos_dist + TRIPLET_MARGIN)
                    
                    if semi_hard_mask.any():
                        semi_hard_indices = torch.where(semi_hard_mask)[0]
                        neg_idx = semi_hard_indices[torch.randint(len(semi_hard_indices), (1,))].item()
                    else:
                        # Fallback to hardest negative
                        neg_idx = anchor_neg_dists.argmin().item()
                
                negative_embed = neg_embeds[neg_idx:neg_idx+1]
                negative_adapted = adapter(negative_embed)
                
                # Compute loss
                loss = triplet_loss(anchor_adapted, positive_adapted, negative_adapted)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_triplets += 1
            
            avg_loss = total_loss / max(n_triplets, 1)
            
            if progress_callback:
                progress_callback(epoch + 1, ADAPTER_EPOCHS, avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Save adapter to profile
        adapter.eval()
        user_profile.adapter = adapter
        user_profile.adapter_trained = True
        
        # Update centroid with adapter
        with torch.no_grad():
            adapted_embeds = adapter(pos_embeds)
            user_profile.centroid = adapted_embeds.mean(dim=0).cpu().numpy()
        
        return True


# =============================================================================
# Authentication Engine
# =============================================================================

class AuthEngine:
    """Core authentication logic with encoder + optional adapter."""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load encoder
        self.encoder = KeystrokeEncoder().to(self.device)
        if ENCODER_PATH.exists():
            state_dict = torch.load(ENCODER_PATH, map_location=self.device)
            self.encoder.load_state_dict(state_dict)
            print(f"Loaded encoder from {ENCODER_PATH}")
        else:
            print(f"Warning: Encoder not found at {ENCODER_PATH}")
        self.encoder.eval()
        
        # Load scaler (optional)
        self.scaler = None
        if SCALER_PATH.exists():
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Loaded scaler from {SCALER_PATH}")
        
        # Load CMU data
        self.cmu_loader = CMUDataLoader(CMU_DATA_PATH, self.scaler)
        
        # Adapter trainer
        self.adapter_trainer = AdapterTrainer(self.encoder, self.cmu_loader, self.device)
        
        # Current user profile
        self.current_profile = None
        
    def sequence_to_embedding(self, sequence, adapter=None):
        """Convert keystroke sequence to embedding."""
        # sequence: (11, 3) numpy array
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embed = self.encoder(x)
            if adapter is not None:
                adapter = adapter.to(self.device)
                embed = adapter(embed)
        
        return embed.cpu().numpy().squeeze()
    
    def compute_similarity(self, embed1, embed2):
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2) + 1e-8))
    
    def enroll_user(self, username, sequences, progress_callback=None):
        """Enroll a new user with multiple keystroke samples."""
        profile = UserProfile(username)
        profile.enrollment_sequences = sequences
        
        # Compute raw embeddings (before adapter)
        embeddings = []
        for seq in sequences:
            embed = self.sequence_to_embedding(seq, adapter=None)
            embeddings.append(embed)
        profile.enrollment_embeddings = embeddings
        profile.raw_centroid = np.mean(embeddings, axis=0)
        profile.centroid = profile.raw_centroid.copy()  # Will be updated after adapter training
        
        # Compute timing statistics
        profile.compute_timing_stats()
        
        # Train adapter with triplet loss
        if len(sequences) >= 2:
            print("Training adapter with triplet loss...")
            success = self.adapter_trainer.train(profile, progress_callback)
            if success:
                print("Adapter trained successfully")
            else:
                print("Adapter training failed, using raw embeddings")
        
        profile.save()
        self.current_profile = profile
        return profile
    
    def load_user(self, username):
        """Load existing user profile."""
        profile = UserProfile(username)
        if profile.load():
            self.current_profile = profile
            return profile
        return None
    
    def verify(self, sequence, profile=None):
        """Verify a login attempt against user profile."""
        if profile is None:
            profile = self.current_profile
        if profile is None:
            return None, 0.0, "No user loaded"
        
        # Get embedding with adapter if available
        adapter = profile.adapter if profile.adapter_trained else None
        embed = self.sequence_to_embedding(sequence, adapter)
        
        # Also get raw embedding for comparison
        raw_embed = self.sequence_to_embedding(sequence, adapter=None)
        
        # Compute similarities
        similarity = self.compute_similarity(embed, profile.centroid)
        raw_similarity = self.compute_similarity(raw_embed, profile.raw_centroid) if profile.raw_centroid is not None else similarity
        
        is_authentic = similarity >= profile.threshold
        
        return is_authentic, similarity, raw_similarity, embed, raw_embed, sequence
    
    def maybe_merge_login(self, profile, similarity, embed, sequence):
        """Merge successful high-confidence login into profile using EMA."""
        if profile is None:
            return False
        
        # Only merge if significantly above threshold
        if similarity < profile.threshold + MERGE_MARGIN:
            return False
        
        profile.successful_logins.append({
            'embedding': embed.tolist(),
            'sequence': sequence.tolist(),
            'similarity': similarity,
            'timestamp': time.time()
        })
        
        # EMA update centroid
        if len(profile.successful_logins) >= MIN_SAMPLES_FOR_MERGE:
            old_centroid = profile.centroid.copy()
            profile.centroid = (1 - EMA_MOMENTUM) * profile.centroid + EMA_MOMENTUM * embed
            profile.centroid = profile.centroid / (np.linalg.norm(profile.centroid) + 1e-8)  # Re-normalize
            profile.merge_count += 1
            profile.save()
            print(f"Merged login into profile (EMA). Total merges: {profile.merge_count}")
            return True
        
        return False


# =============================================================================
# Keystroke Capture
# =============================================================================

class KeystrokeCapture:
    """Real-time keystroke timing capture."""
    
    def __init__(self):
        self.reset()
        
        # Map special keys
        self.key_map = {
            '.': 'period',
            '5': 'five',
            '\r': 'Return',
            'Return': 'Return',
        }
        
    def reset(self):
        self.key_events = []  # [(key_name, press_time, release_time), ...]
        self.pending_keys = {}  # {key_name: press_time}
        self.last_key_down_time = None
        self.last_key_up_time = None
        
    def normalize_key(self, char, is_shift=False):
        """Normalize key name to match expected format."""
        if char in self.key_map:
            return self.key_map[char]
        if is_shift and char.lower() == 'r':
            return 'Shift.r'
        return char.lower()
    
    def key_press(self, char, is_shift=False):
        """Record key press event."""
        key = self.normalize_key(char, is_shift)
        current_time = time.time()
        
        if key not in self.pending_keys:
            self.pending_keys[key] = current_time
            self.last_key_down_time = current_time
    
    def key_release(self, char, is_shift=False):
        """Record key release event."""
        key = self.normalize_key(char, is_shift)
        current_time = time.time()
        
        if key in self.pending_keys:
            press_time = self.pending_keys.pop(key)
            self.key_events.append((key, press_time, current_time))
            self.last_key_up_time = current_time
    
    def get_sequence(self):
        """Convert captured events to feature sequence (11, 3)."""
        # Build timing features for each expected key
        sequence = []
        
        # Create lookup for captured events
        event_lookup = {event[0]: event for event in self.key_events}
        
        prev_down_time = None
        prev_up_time = None
        
        for i, key in enumerate(KEY_ORDER):
            if key in event_lookup:
                _, press_time, release_time = event_lookup[key]
                
                # H: Hold time (release - press)
                h = release_time - press_time
                
                # DD: Down-to-down (current press - previous press)
                if prev_down_time is not None:
                    dd = press_time - prev_down_time
                else:
                    dd = 0.0
                
                # UD: Up-to-down (current press - previous release)
                if prev_up_time is not None:
                    ud = press_time - prev_up_time
                else:
                    ud = 0.0
                
                prev_down_time = press_time
                prev_up_time = release_time
            else:
                # Key not captured, use zeros
                h = 0.0
                dd = 0.0
                ud = 0.0
            
            sequence.append([h, dd, ud])
        
        return np.array(sequence, dtype=np.float32)
    
    def is_complete(self):
        """Check if all expected keys have been captured."""
        captured_keys = {event[0] for event in self.key_events}
        expected_keys = set(KEY_ORDER)
        return expected_keys.issubset(captured_keys)


# =============================================================================
# Visualization Functions
# =============================================================================

def create_timing_figure(profile, current_sequence=None):
    """Create timing visualization (H, DD, UD bar plots)."""
    fig = Figure(figsize=(10, 8), dpi=100)
    
    colors = ['#2ecc71', '#3498db', '#9b59b6']  # Green, Blue, Purple
    feature_names = ['H (Hold)', 'DD (Down-Down)', 'UD (Up-Down)']
    
    for i, (fname, color) in enumerate(zip(feature_names, colors)):
        ax = fig.add_subplot(3, 1, i + 1)
        
        short_name = fname.split()[0]  # 'H', 'DD', 'UD'
        
        x = np.arange(len(KEY_ORDER))
        width = 0.35
        
        # Enrollment average
        if profile and profile.timing_stats[short_name]['mean']:
            means = profile.timing_stats[short_name]['mean']
            stds = profile.timing_stats[short_name]['std']
            ax.bar(x - width/2, means, width, label='Enrollment Avg', color=color, alpha=0.7)
            ax.errorbar(x - width/2, means, yerr=stds, fmt='none', color='black', capsize=3)
        
        # Current sequence
        if current_sequence is not None:
            current_values = current_sequence[:, i]
            ax.bar(x + width/2, current_values, width, label='Current', color=color, alpha=0.4, hatch='//')
        
        ax.set_ylabel(f'{fname} (s)')
        ax.set_xticks(x)
        ax.set_xticklabels(KEY_ORDER, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(fname)
        ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    return fig


def create_comparison_figure(profile, current_sequence):
    """Create enrollment vs current login comparison."""
    fig = Figure(figsize=(10, 6), dpi=100)
    
    # Heatmap comparison
    ax1 = fig.add_subplot(1, 2, 1)
    
    if profile and profile.enrollment_sequences:
        enrollment_avg = np.mean(np.stack(profile.enrollment_sequences), axis=0)
        diff = np.abs(current_sequence - enrollment_avg)
        im = ax1.imshow(diff.T, aspect='auto', cmap='RdYlGn_r')
        ax1.set_xticks(range(len(KEY_ORDER)))
        ax1.set_xticklabels(KEY_ORDER, rotation=45, ha='right', fontsize=8)
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(['H', 'DD', 'UD'])
        ax1.set_title('Timing Difference (Enrollment vs Current)')
        ax1.set_xlabel('Keys')
        ax1.set_ylabel('Features')
        fig.colorbar(im, ax=ax1, label='Abs Difference (s)')
    
    # Per-feature deviation
    ax2 = fig.add_subplot(1, 2, 2)
    
    if profile and profile.enrollment_sequences:
        enrollment_stack = np.stack(profile.enrollment_sequences)
        enrollment_std = enrollment_stack.std(axis=0)  # (11, 3)
        
        deviations = []
        labels = []
        for i, fname in enumerate(['H', 'DD', 'UD']):
            # How many std deviations is current from mean?
            mean_vals = enrollment_stack[:, :, i].mean(axis=0)
            std_vals = enrollment_std[:, i] + 1e-6
            current_vals = current_sequence[:, i]
            z_scores = np.abs((current_vals - mean_vals) / std_vals)
            deviations.append(z_scores.mean())
            labels.append(fname)
        
        bars = ax2.bar(labels, deviations, color=['#2ecc71', '#3498db', '#9b59b6'])
        ax2.axhline(y=2, color='orange', linestyle='--', label='2σ threshold')
        ax2.axhline(y=3, color='red', linestyle='--', label='3σ threshold')
        ax2.set_ylabel('Avg Z-Score')
        ax2.set_title('Feature Deviation from Enrollment')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    return fig


def create_similarity_figure(similarity_history, threshold):
    """Create similarity score history plot."""
    fig = Figure(figsize=(8, 4), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    
    if similarity_history:
        x = range(1, len(similarity_history) + 1)
        colors = ['green' if s >= threshold else 'red' for s in similarity_history]
        ax.bar(x, similarity_history, color=colors, alpha=0.7)
        ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
        ax.set_xlabel('Login Attempt')
        ax.set_ylabel('Similarity Score')
        ax.set_title('Login Similarity History')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No login attempts yet', ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    fig.tight_layout()
    return fig


def create_adapter_effect_figure(raw_sim, adapted_sim, threshold):
    """Show effect of adapter on similarity."""
    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    
    x = ['Raw Encoder', 'With Adapter']
    values = [raw_sim, adapted_sim]
    colors = ['#3498db', '#9b59b6']
    
    bars = ax.bar(x, values, color=colors, alpha=0.7)
    ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Adapter Effect on Similarity')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    return fig


# =============================================================================
# Main Application
# =============================================================================

class KeystrokeAuthApp:
    """Main Tkinter application."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Keystroke Dynamics Authentication (Research Grade)")
        self.root.geometry("1200x900")
        
        # Initialize engine
        self.engine = AuthEngine()
        self.capture = KeystrokeCapture()
        
        # State
        self.mode = None  # 'enroll' or 'login'
        self.enrollment_samples = []
        self.similarity_history = []
        self.current_sequence = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Main container with notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Authentication
        self.auth_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.auth_frame, text='Authentication')
        self.setup_auth_tab()
        
        # Tab 2: Timing Visualization
        self.timing_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.timing_frame, text='Timing Analysis')
        self.setup_timing_tab()
        
        # Tab 3: Comparison
        self.compare_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.compare_frame, text='Comparison')
        self.setup_compare_tab()
        
        # Tab 4: History
        self.history_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.history_frame, text='Login History')
        self.setup_history_tab()
        
    def setup_auth_tab(self):
        """Setup authentication tab."""
        # Top frame for user selection
        top_frame = ttk.Frame(self.auth_frame)
        top_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(top_frame, text="Username:", font=('Arial', 12)).pack(side='left', padx=5)
        self.username_var = tk.StringVar()
        self.username_entry = ttk.Entry(top_frame, textvariable=self.username_var, width=20, font=('Arial', 12))
        self.username_entry.pack(side='left', padx=5)
        
        ttk.Button(top_frame, text="Enroll New User", command=self.start_enrollment).pack(side='left', padx=10)
        ttk.Button(top_frame, text="Login", command=self.start_login).pack(side='left', padx=10)
        ttk.Button(top_frame, text="Load User", command=self.load_user).pack(side='left', padx=10)
        
        # Password display
        pw_frame = ttk.Frame(self.auth_frame)
        pw_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(pw_frame, text=f"Password to type: ", font=('Arial', 11)).pack(side='left')
        ttk.Label(pw_frame, text=PASSWORD, font=('Courier', 14, 'bold'), foreground='blue').pack(side='left')
        
        # Status label
        self.status_var = tk.StringVar(value="Enter username and choose Enroll or Login")
        self.status_label = ttk.Label(self.auth_frame, textvariable=self.status_var, 
                                       font=('Arial', 12), wraplength=600)
        self.status_label.pack(pady=10)
        
        # Password input frame
        input_frame = ttk.LabelFrame(self.auth_frame, text="Type Password Here", padding=10)
        input_frame.pack(fill='x', padx=10, pady=10)
        
        self.password_entry = tk.Entry(input_frame, font=('Courier', 16), show='*', width=30)
        self.password_entry.pack(pady=10)
        self.password_entry.bind('<KeyPress>', self.on_key_press)
        self.password_entry.bind('<KeyRelease>', self.on_key_release)
        
        # Progress bar for enrollment
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.auth_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', padx=10, pady=5)
        
        # Progress label
        self.progress_label = tk.StringVar(value="")
        ttk.Label(self.auth_frame, textvariable=self.progress_label, font=('Arial', 10)).pack()
        
        # Result frame
        result_frame = ttk.LabelFrame(self.auth_frame, text="Result", padding=10)
        result_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.result_text = tk.Text(result_frame, height=10, font=('Courier', 11), state='disabled')
        self.result_text.pack(fill='both', expand=True)
        
        # Threshold adjustment
        threshold_frame = ttk.Frame(self.auth_frame)
        threshold_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(threshold_frame, text="Threshold:", font=('Arial', 10)).pack(side='left')
        self.threshold_var = tk.DoubleVar(value=DEFAULT_THRESHOLD)
        self.threshold_scale = ttk.Scale(threshold_frame, from_=0.5, to=0.95, 
                                          variable=self.threshold_var, orient='horizontal', length=200)
        self.threshold_scale.pack(side='left', padx=10)
        self.threshold_display = ttk.Label(threshold_frame, text=f"{DEFAULT_THRESHOLD:.2f}", font=('Arial', 10))
        self.threshold_display.pack(side='left')
        self.threshold_scale.bind('<Motion>', self.update_threshold_display)
        
        # Merge option
        self.auto_merge_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(threshold_frame, text="Auto-merge high-confidence logins", 
                        variable=self.auto_merge_var).pack(side='right', padx=10)
        
    def setup_timing_tab(self):
        """Setup timing visualization tab."""
        self.timing_canvas_frame = ttk.Frame(self.timing_frame)
        self.timing_canvas_frame.pack(fill='both', expand=True)
        
        # Placeholder
        self.timing_fig = None
        self.timing_canvas = None
        
    def setup_compare_tab(self):
        """Setup comparison tab."""
        self.compare_canvas_frame = ttk.Frame(self.compare_frame)
        self.compare_canvas_frame.pack(fill='both', expand=True)
        
        self.compare_fig = None
        self.compare_canvas = None
        
    def setup_history_tab(self):
        """Setup login history tab."""
        self.history_canvas_frame = ttk.Frame(self.history_frame)
        self.history_canvas_frame.pack(fill='both', expand=True)
        
        self.history_fig = None
        self.history_canvas = None
        
    def update_threshold_display(self, event=None):
        """Update threshold display label."""
        val = self.threshold_var.get()
        self.threshold_display.config(text=f"{val:.2f}")
        if self.engine.current_profile:
            self.engine.current_profile.threshold = val
            
    def log_result(self, message):
        """Log message to result text widget."""
        self.result_text.config(state='normal')
        self.result_text.insert('end', message + '\n')
        self.result_text.see('end')
        self.result_text.config(state='disabled')
        
    def clear_result(self):
        """Clear result text widget."""
        self.result_text.config(state='normal')
        self.result_text.delete('1.0', 'end')
        self.result_text.config(state='disabled')
        
    def start_enrollment(self):
        """Start enrollment mode."""
        username = self.username_var.get().strip()
        if not username:
            messagebox.showerror("Error", "Please enter a username")
            return
        
        self.mode = 'enroll'
        self.enrollment_samples = []
        self.capture.reset()
        self.password_entry.delete(0, 'end')
        self.clear_result()
        
        self.status_var.set(f"Enrollment mode for '{username}'. Type the password {MIN_ENROLLMENT_SAMPLES} times.\nSample 1/{MIN_ENROLLMENT_SAMPLES}")
        self.progress_var.set(0)
        self.progress_label.set("")
        self.password_entry.focus()
        
    def start_login(self):
        """Start login mode."""
        username = self.username_var.get().strip()
        if not username:
            messagebox.showerror("Error", "Please enter a username")
            return
        
        profile = self.engine.load_user(username)
        if profile is None:
            messagebox.showerror("Error", f"User '{username}' not found. Please enroll first.")
            return
        
        self.mode = 'login'
        self.capture.reset()
        self.password_entry.delete(0, 'end')
        self.clear_result()
        
        self.threshold_var.set(profile.threshold)
        self.update_threshold_display()
        
        adapter_status = "with trained adapter" if profile.adapter_trained else "raw encoder only"
        self.status_var.set(f"Login mode for '{username}' ({adapter_status}). Type the password.")
        self.progress_var.set(100)
        self.password_entry.focus()
        
    def load_user(self):
        """Load existing user profile."""
        username = self.username_var.get().strip()
        if not username:
            messagebox.showerror("Error", "Please enter a username")
            return
        
        profile = self.engine.load_user(username)
        if profile:
            self.threshold_var.set(profile.threshold)
            self.update_threshold_display()
            
            adapter_status = "Adapter trained" if profile.adapter_trained else "No adapter"
            self.status_var.set(f"Loaded user '{username}'. {adapter_status}. {len(profile.enrollment_sequences)} enrollment samples.")
            self.log_result(f"User '{username}' loaded successfully")
            self.log_result(f"  - Enrollment samples: {len(profile.enrollment_sequences)}")
            self.log_result(f"  - Adapter trained: {profile.adapter_trained}")
            self.log_result(f"  - Login count: {profile.login_count}")
            self.log_result(f"  - Merge count: {profile.merge_count}")
            
            # Update visualizations
            self.update_timing_visualization()
        else:
            messagebox.showerror("Error", f"User '{username}' not found")
            
    def on_key_press(self, event):
        """Handle key press event."""
        char = event.char
        keysym = event.keysym
        
        # Detect Shift+R
        is_shift = bool(event.state & 0x1)
        
        if keysym == 'Return':
            self.capture.key_press('Return')
        elif keysym in ('Shift_L', 'Shift_R'):
            pass  # Don't capture shift alone
        elif char:
            self.capture.key_press(char, is_shift)
            
    def on_key_release(self, event):
        """Handle key release event."""
        char = event.char
        keysym = event.keysym
        
        is_shift = bool(event.state & 0x1)
        
        if keysym == 'Return':
            self.capture.key_release('Return')
            self.root.after(100, self.process_input)
        elif keysym in ('Shift_L', 'Shift_R'):
            pass
        elif char:
            self.capture.key_release(char, is_shift)
            
    def process_input(self):
        """Process captured keystroke input."""
        typed = self.password_entry.get()
        
        if typed != PASSWORD:
            self.log_result(f"Incorrect password typed: '{typed}'")
            self.password_entry.delete(0, 'end')
            self.capture.reset()
            return
        
        # Get keystroke sequence
        sequence = self.capture.get_sequence()
        self.current_sequence = sequence
        
        if self.mode == 'enroll':
            self.process_enrollment(sequence)
        elif self.mode == 'login':
            self.process_login(sequence)
        
        # Reset for next input
        self.password_entry.delete(0, 'end')
        self.capture.reset()
        
    def process_enrollment(self, sequence):
        """Process enrollment sample."""
        self.enrollment_samples.append(sequence)
        count = len(self.enrollment_samples)
        
        self.log_result(f"Sample {count} captured")
        
        if count < MIN_ENROLLMENT_SAMPLES:
            progress = (count / MIN_ENROLLMENT_SAMPLES) * 100
            self.progress_var.set(progress)
            self.status_var.set(f"Sample {count + 1}/{MIN_ENROLLMENT_SAMPLES}")
        else:
            self.progress_var.set(50)
            self.status_var.set("Processing enrollment...")
            self.progress_label.set("Training adapter...")
            self.root.update()
            
            # Train in thread to keep UI responsive
            def train_callback(epoch, total, loss):
                progress = 50 + (epoch / total) * 50
                self.progress_var.set(progress)
                self.progress_label.set(f"Training adapter: epoch {epoch}/{total}, loss={loss:.4f}")
                self.root.update()
            
            username = self.username_var.get().strip()
            profile = self.engine.enroll_user(username, self.enrollment_samples, train_callback)
            
            self.progress_var.set(100)
            self.progress_label.set("Enrollment complete!")
            self.status_var.set(f"User '{username}' enrolled successfully!")
            
            self.log_result(f"\nEnrollment complete for '{username}'")
            self.log_result(f"  - Samples: {len(profile.enrollment_sequences)}")
            self.log_result(f"  - Adapter trained: {profile.adapter_trained}")
            self.log_result(f"  - Threshold: {profile.threshold:.2f}")
            
            # Update visualizations
            self.update_timing_visualization()
            
            self.mode = None
            
    def process_login(self, sequence):
        """Process login attempt."""
        profile = self.engine.current_profile
        result = self.engine.verify(sequence, profile)
        
        if result is None:
            self.log_result("No user loaded!")
            return
        
        is_authentic, similarity, raw_similarity, embed, raw_embed, seq = result
        
        profile.login_count += 1
        self.similarity_history.append(similarity)
        
        # Log result
        status = "✓ AUTHENTICATED" if is_authentic else "✗ REJECTED"
        color = "green" if is_authentic else "red"
        
        self.log_result(f"\n{'='*50}")
        self.log_result(f"Login attempt #{profile.login_count}: {status}")
        self.log_result(f"  Similarity (with adapter): {similarity:.4f}")
        self.log_result(f"  Similarity (raw encoder):  {raw_similarity:.4f}")
        self.log_result(f"  Threshold: {profile.threshold:.2f}")
        
        # Maybe merge
        if is_authentic and self.auto_merge_var.get():
            merged = self.engine.maybe_merge_login(profile, similarity, embed, sequence)
            if merged:
                self.log_result(f"  → Profile updated via EMA (merge #{profile.merge_count})")
        
        profile.save()
        
        # Update visualizations
        self.update_timing_visualization()
        self.update_compare_visualization()
        self.update_history_visualization()
        
        # Show adapter effect tab
        self.update_adapter_effect_visualization(raw_similarity, similarity)
        
    def update_timing_visualization(self):
        """Update timing analysis visualization."""
        profile = self.engine.current_profile
        if profile is None:
            return
        
        # Clear previous
        for widget in self.timing_canvas_frame.winfo_children():
            widget.destroy()
        
        fig = create_timing_figure(profile, self.current_sequence)
        canvas = FigureCanvasTkAgg(fig, self.timing_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self.timing_canvas = canvas
        
    def update_compare_visualization(self):
        """Update comparison visualization."""
        profile = self.engine.current_profile
        if profile is None or self.current_sequence is None:
            return
        
        for widget in self.compare_canvas_frame.winfo_children():
            widget.destroy()
        
        fig = create_comparison_figure(profile, self.current_sequence)
        canvas = FigureCanvasTkAgg(fig, self.compare_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self.compare_canvas = canvas
        
    def update_history_visualization(self):
        """Update login history visualization."""
        profile = self.engine.current_profile
        if profile is None:
            return
        
        for widget in self.history_canvas_frame.winfo_children():
            widget.destroy()
        
        fig = create_similarity_figure(self.similarity_history, profile.threshold)
        canvas = FigureCanvasTkAgg(fig, self.history_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self.history_canvas = canvas
        
    def update_adapter_effect_visualization(self, raw_sim, adapted_sim):
        """Show adapter effect in a popup or dedicated area."""
        profile = self.engine.current_profile
        if profile is None:
            return
        
        # Create popup window for adapter effect
        popup = tk.Toplevel(self.root)
        popup.title("Adapter Effect")
        popup.geometry("500x400")
        
        fig = create_adapter_effect_figure(raw_sim, adapted_sim, profile.threshold)
        canvas = FigureCanvasTkAgg(fig, popup)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=10)


# =============================================================================
# Entry Point
# =============================================================================

def main():
    root = tk.Tk()
    app = KeystrokeAuthApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
