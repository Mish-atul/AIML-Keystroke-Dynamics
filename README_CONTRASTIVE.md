# Keystroke Dynamics: Contrastive Encoder + Per-User Adapters

A research and deployment pipeline for keystroke dynamics authentication using contrastive learning and few-shot per-user adapters.

## Overview

This project implements:
1. **Global Contrastive Encoder**: A 1D-CNN trained with supervised contrastive loss (SupCon) on the CMU Keystroke Benchmark
2. **Per-User Adapters**: Small MLPs (<10k params) for few-shot personalization (1-10 enrollment samples)
3. **Evaluation Framework**: Comparison against RF, XGBoost, HGBT, MLP, CNN baselines
4. **Deployment**: REST API (FastAPI) and Streamlit demo

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_contrastive.txt
```

### 2. Prepare Dataset

The CMU Keystroke Dynamics Benchmark dataset should be placed at:
```
model training/DSL-StrongPasswordData.csv
```

It will be automatically copied to `cmu_keystroke.csv` on first run.

### 3. Run Smoke Tests

```bash
python scripts/run_smoke_tests.py
```

### 4. Train Encoder

```bash
python scripts/train_encoder.py --csv cmu_keystroke.csv --epochs 120 --batch-size 128 --out artifacts/encoder.pt
```

### 5. Train Adapters

For all users:
```bash
python scripts/train_adapter.py --encoder artifacts/encoder.pt --all --shots 5
```

For a single user:
```bash
python scripts/train_adapter.py --encoder artifacts/encoder.pt --user s002 --shots 5
```

### 6. Evaluate

```bash
python scripts/evaluate.py --encoder artifacts/encoder.pt --split both --baselines --out results/
```

### 7. Run Demo

Streamlit:
```bash
streamlit run streamlit_app/app.py
```

REST API:
```bash
cd api && uvicorn app:app --reload --port 8000
```

## Canonical Sequence Format

**Shape: (N, 11, 3)**

- N = number of samples
- 11 = key tokens in password `.tie5Roanl`
- 3 = features per key: [H (hold), DD (down-to-down), UD (up-to-down)]

Key order:
```
period, t, i, e, five, Shift.r, o, a, n, l, Return
```

## Project Structure

```
keystroke/
├── contrastive_encoder/          # Main package
│   ├── config.py                 # Hyperparameters
│   ├── data/                     # Preprocessing, datasets, augmentations
│   ├── models/                   # Encoder, adapter, losses
│   ├── training/                 # Training loops
│   ├── evaluation/               # Metrics, visualization
│   └── utils/                    # Helpers
├── scripts/                      # CLI entry points
│   ├── train_encoder.py
│   ├── train_adapter.py
│   ├── evaluate.py
│   ├── enroll_user.py
│   ├── verify_user.py
│   └── run_smoke_tests.py
├── api/                          # FastAPI REST API
│   └── app.py
├── streamlit_app/                # Streamlit demo
│   └── app.py
├── artifacts/                    # Trained models (generated)
│   ├── encoder.pt
│   ├── adapters/
│   ├── templates/
│   └── checkpoints/
├── results/                      # Evaluation outputs (generated)
├── cmu_keystroke.csv             # Dataset (copied)
├── data_manifest.json            # Dataset metadata
└── requirements_contrastive.txt
```

## Hyperparameters

### Encoder

| Parameter | Default | Description |
|-----------|---------|-------------|
| embedding_dim | 128 | Output embedding dimension |
| batch_size | 128 | Training batch size |
| epochs | 120 | Training epochs |
| learning_rate | 1e-3 | AdamW learning rate |
| temperature | 0.07 | Contrastive temperature (τ) |
| use_supcon | True | Use supervised contrastive loss |

### Adapter

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_dim | 64 | Hidden layer dimension |
| dropout | 0.2 | Dropout rate |
| max_steps | 200 | Training steps |
| loss_type | triplet | Loss function |

### Augmentations

| Augmentation | Probability | Scale |
|--------------|-------------|-------|
| Gaussian Jitter | 0.5 | ±3% |
| Time Warp | 0.2 | ±10% |
| Key Dropout | 0.02 | - |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/enroll` | POST | Enroll new user with samples |
| `/verify` | POST | Verify sample against user |
| `/users` | GET | List enrolled users |
| `/health` | GET | Health check |

Example:
```bash
# Enroll
curl -X POST http://localhost:8000/enroll \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice", "samples": [...]}'

# Verify
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice", "sample": {...}}'
```

## Expected Results

Based on the CMU Keystroke Dynamics Benchmark (51 users, 20,400 samples):

| Model | Accuracy | AUC | EER |
|-------|----------|-----|-----|
| XGBoost (baseline) | 94.53% | 0.9475 | 12.07% |
| Encoder (no adapter) | ~92% | ~0.94 | ~13% |
| Encoder + Adapter (5-shot) | ~93% | ~0.95 | ~11% |

## Hardware Requirements

- **Training**: GPU recommended (RTX 2080 or better), ~2-6 hours
- **Inference**: CPU sufficient, <10ms per verification
- **Adapter training**: CPU ok, <2 minutes per user

## Citation

If you use this code, please cite:

```
CMU Keystroke Dynamics Benchmark:
Killourhy, K.S., Maxion, R.A. (2009). Comparing anomaly-detection 
algorithms for keystroke dynamics. DSN 2009.

Contrastive Learning:
Chen et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations.
Khosla et al. (2020). Supervised Contrastive Learning. NeurIPS 2020.
```

## License

Research use only. See dataset license for CMU benchmark data.
