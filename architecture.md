# Architecture

This project has two layers: (1) an ML/DL training pipeline that produces keystroke-dynamics models and artifacts, and (2) two Tkinter desktop apps that consume those artifacts for authentication.

## Component overview

- **Data**: CMU Keystroke Dynamics Benchmark (`DSL-StrongPasswordData.csv`) with fixed password `.tie5Roanl` (51 users, 20,400 samples).
- **Training pipeline** (`model training/keystroke_training_pipeline.py`):
  - Loads CSV, performs feature engineering (dwell, flight, digraph latencies + statistical aggregates) -> 47-feature vectors; also prepares sequences for CNN.
  - Trains 5 models: Random Forest, XGBoost, Histogram Gradient Boosting, MLP, 1D CNN.
  - Evaluates identification (accuracy, confusion matrices) and verification (ROC, AUC, FAR/FRR, EER), generates comparison tables and plots.
  - Persists models (`rf_model.pkl`, `xgb_model.json`, `hgb_model.pkl`, `mlp_model.h5`, `cnn_model.h5`), `scaler.pkl`, `label_encoder.pkl`, plus visualizations and CSVs into organized folders.
- **Desktop apps**:
  - `app/keystroke_auth_app.py`: any-password variant. Captures keystroke events, extracts 47-dim features, scales, compares to user template, and checks RF prediction. User profiles stored in `users/<username>.json` with mean vector, threshold, RF label, plaintext password (prototype only).
  - `higher_acc_app/keystroke_fixed_password_app.py`: fixed-password variant (`.tie5Roanl`) for tighter distributions and higher accuracy. Same hybrid decision (template distance + RF label) and JSON storage.

## Data flow

1. **Training**
   - Input: `DSL-StrongPasswordData.csv`.
   - Processing: feature engineering -> train/test splits -> scaling -> model training (5 models) -> evaluation -> artifact generation.
   - Outputs: models, scaler, label encoder, plots (confusion matrices, feature importance, training curves, ROC, distance), `model_comparison_table.csv`.

2. **App runtime (registration)**
   - User enters username + password (any-password app) or fixed password (higher-acc app).
   - Keystrokes captured (down/up with timestamps) -> 47-dim features -> scaled with `scaler.pkl` -> RF model predicts style label.
   - Build template: mean feature vector over 5 attempts, threshold = mean distance + 2*std, most common RF label; store JSON under `users/`.

3. **App runtime (login)**
   - User provides credentials; keystrokes captured -> features -> scaled.
   - Compute Euclidean distance to stored mean vector; compare to threshold.
   - RF predicts style label; hybrid decision = (distance < threshold) AND RF label match AND password match.

## Dependencies

- Training: Python 3, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `tensorflow` (see `model training/requirements.txt`).
- Apps: Python with `tkinter` plus `numpy`, `joblib`; expects trained artifacts in the same folder as the app script.

## Security considerations

- Passwords are stored in plaintext JSON for demo purposes; production should add hashing, secure storage, rate limiting, MFA, and hardened model handling.

## Deployment/usage notes

- To retrain: run the pipeline; copy refreshed `rf_model.pkl`, `scaler.pkl`, and optional `label_encoder.pkl` into `app/` or `higher_acc_app/`.
- Binary model files are large; keep them co-located with the apps for offline use.
