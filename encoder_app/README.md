# Encoder-Based Keystroke Authentication App

This desktop application uses the **Contrastive Encoder + Per-User Adapter** system for keystroke dynamics authentication with **proper triplet loss training** and **real CMU negative samples**.

## Features

- **Real-time keystroke capture** using Tkinter
- **1D-CNN contrastive encoder** for embeddings (128-dimensional)
- **Per-user adapter training** with triplet loss and semi-hard negative mining
- **Real CMU negative samples** (51 users, 20,400 samples)
- **Backend visualizations**: H/DD/UD timing plots, enrollment vs login comparison
- **Continuous learning**: EMA centroid update for high-confidence logins
- **Cosine similarity verification** against user centroid

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- tkinter (usually comes with Python)

## Usage

```bash
cd encoder_app
python keystroke_auth_app.py
```

## How It Works

### Registration
1. Enter a username
2. Click "Enroll New User"
3. Type the password `.tie5Roanl` **5 times** (press Enter after each)
4. The app trains a personal adapter using triplet loss with CMU negatives
5. Your typing pattern (centroid + adapter) is saved

### Login
1. Enter your username
2. Click "Login"
3. Type the password `.tie5Roanl`
4. The app extracts keystroke features, passes through encoder + your adapter
5. Compares with your stored centroid using cosine similarity
6. Accepts if similarity >= threshold (default: 0.85)

### Visualizations
- **Timing Analysis**: Bar plots of H (hold), DD (down-down), UD (up-down) for each key
- **Comparison**: Heatmap showing difference between enrollment and current login
- **Login History**: Similarity scores for all login attempts
- **Adapter Effect**: Shows raw encoder vs adapter similarity

## Technical Details

### Feature Format
- Input: 11 keys × 3 features = (11, 3) sequence
- Features per key:
  - H (hold time): key down to key up
  - DD (down-to-down): time from previous key down to current key down
  - UD (up-to-down): time from previous key up to current key down

### Encoder
- 1D-CNN architecture
- Trained with Supervised Contrastive Loss
- Output: 128-dimensional L2-normalized embedding

### Adapter
- Small MLP (~17k parameters per user)
- Trained with few-shot enrollment samples
- Personalizes the global encoder for each user

## Files

- `keystroke_encoder_app.py` - Main application
- `users/` - Directory for user profiles and adapters
  - `{username}.json` - User profile (centroid, threshold)
  - `{username}_adapter.pt` - Trained adapter weights

## Difference from RF-based App

| Feature | RF App (`app/`) | Encoder App (`encoder_app/`) |
|---------|----------------|------------------------------|
| Model | Random Forest + Template | 1D-CNN Encoder + Adapter |
| Features | 47-dim aggregated | (11, 3) sequence |
| Training | Pre-trained on all users | Per-user adapter |
| Personalization | None | Few-shot adapter |
| Embedding | None | 128-dim learned |
