# AIML Keystroke Dynamics

Keystroke-dynamics authentication project with a full ML/DL training pipeline and two Tkinter desktop apps (one any-password, one fixed-password) for user verification. Models and supporting assets are produced from the CMU Keystroke Dynamics Benchmark dataset.

## Repository layout

```
keystroke/
├── model training/           # End-to-end training pipeline and artifacts
│   ├── keystroke_training_pipeline.py
│   ├── requirements.txt
│   ├── DSL-StrongPasswordData.csv     # CMU dataset (.tie5Roanl)
│   ├── models/                        # Trained models (RF, XGB, HGBT, MLP, CNN, scaler, label encoder)
│   ├── confusion_matrices/            # 5 images
│   ├── feature_importance/            # 2 images
│   ├── training_history/              # 2 images
│   ├── verification/                  # ROC + distance plots
│   ├── results/                       # model_comparison_table.csv, charts
│   ├── README.md / QUICK_START.md / STRUCTURE.md / RESULTS_SUMMARY.md
│   └── organize_files.py
├── app/                    # Desktop app (any password) – hybrid template + RF auth
│   ├── keystroke_auth_app.py
│   ├── rf_model.pkl | scaler.pkl | label_encoder.pkl
│   └── users/                 # Stored user profiles (JSON)
├── higher_acc_app/          # Desktop app (fixed password .tie5Roanl) – higher accuracy
│   ├── keystroke_fixed_password_app.py
│   ├── rf_model.pkl | scaler.pkl | label_encoder.pkl
│   └── users/                 # Stored user profiles (JSON)
└── docs/ (planned)
```

## Quick start

### 1) Train models (optional if you use provided artifacts)
```bash
cd "model training"
pip install -r requirements.txt
python keystroke_training_pipeline.py
```
Outputs: models saved to `models/`, visualizations and comparison tables saved under their respective folders.

### 2) Run the desktop app (any-password variant)
```bash
cd app
python keystroke_auth_app.py
```
Requirements: Python with `tkinter` (built-in), `numpy`, `joblib`. Ensure `rf_model.pkl` and `scaler.pkl` sit next to the script; `label_encoder.pkl` is optional. Register with any password (5 attempts) then log in with the same password; hybrid decision uses template distance + RF label.

### 3) Run the desktop app (fixed-password, higher accuracy)
```bash
cd higher_acc_app
python keystroke_fixed_password_app.py
```
Password is fixed to `.tie5Roanl` (matches training data). Same model file placement as above. Register username with 5 attempts, then log in; the fixed password yields tighter feature distributions and more stable RF labels.

## What the training pipeline does
- Loads CMU dataset (`DSL-StrongPasswordData.csv`).
- Feature engineering: dwell times, flight times, digraph latencies, plus statistical aggregates.
- Trains five models: Random Forest, XGBoost, Histogram Gradient Boosting, MLP, and 1D CNN.
- Evaluates identification (accuracy, confusion matrices, reports) and verification (ROC, AUC, FAR/FRR, EER).
- Saves models, scalers, label encoder, plots, and comparison tables.

## Key results (from RESULTS_SUMMARY.md)
- **Best overall:** XGBoost — 94.53% accuracy, AUC 0.9475, EER 12.07%.
- HGBT and RF provide strong performance with faster training; CNN/MLP also >92% accuracy.

## Model assets expected by the apps
- `rf_model.pkl` — RandomForestClassifier trained on 47 features
- `scaler.pkl` — StandardScaler fitted on training data
- `label_encoder.pkl` (optional) — maps numeric labels back to subject IDs

Place these files in the same folder as the app script you run (`app/` or `higher_acc_app/`).

## Dataset
- CMU Keystroke Dynamics Benchmark (password: `.tie5Roanl`), 51 users, 20,400 samples.
- Included locally as `DSL-StrongPasswordData.csv` under `model training/`.

## Troubleshooting
- **Model load errors:** Confirm `rf_model.pkl` and `scaler.pkl` are beside the app script.
- **Login fails often:** Re-register with steadier typing; ensure password consistency; for best accuracy, use the fixed-password app.
- **tkinter missing:** Install a standard Python distribution that bundles Tk (e.g., python.org installer).

## Contributing / notes
- Educational / research prototype; passwords are stored plaintext in app profiles. For production, add hashing, secure storage, rate limiting, and MFA.
- Large binary artifacts (models) are already checked in; keep them co-located for app execution.
