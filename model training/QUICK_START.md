# Keystroke Dynamics Training Pipeline - Quick Start Guide

## ✅ What Has Been Created

### Main Script
- **`keystroke_training_pipeline.py`** - Complete ML/DL training pipeline (870 lines)

### Supporting Files
- **`requirements.txt`** - All required Python packages
- **`README.md`** - Comprehensive documentation

## 🚀 Current Status

The training pipeline is **RUNNING** and will:

1. ✅ Load CMU dataset (20,400 samples, 51 users)
2. ⏳ Extract 31 raw timing features + 18 statistical features = 49 total features
3. ⏳ Train 5 models (RF, XGBoost, HGBT, MLP, CNN)
4. ⏳ Evaluate identification (accuracy, confusion matrix, reports)
5. ⏳ Evaluate verification (ROC, AUC, EER, FAR, FRR)
6. ⏳ Generate comparison table
7. ⏳ Save all models and visualizations

## 📊 Expected Outputs

### Models (7 files)
- `rf_model.pkl`
- `xgb_model.json`
- `hgb_model.pkl`
- `mlp_model.h5`
- `cnn_model.h5`
- `scaler.pkl`
- `label_encoder.pkl`

### Visualizations (~12 files)
- 5 confusion matrices
- 2 feature importance plots
- 2 training history plots
- 1 ROC curve
- 1 distance distribution
- 1 comparison chart

### Tables
- `model_comparison_table.csv`

## ⏱️ Estimated Time
**Total: 5-15 minutes** (depending on hardware)
- Feature extraction: 1-2 min
- RF training: 30s
- XGBoost: 30s
- HGBT: 20s
- MLP: 2-3 min
- CNN: 2-3 min
- Evaluation: 1-2 min

## 🎯 Features Engineered

### Raw Features (31)
1. **Dwell Time (11)**: How long each key is pressed
2. **Flight Time (10)**: Time between releasing & pressing keys  
3. **Digraph Latency (10)**: Time between consecutive key presses

### Statistical Features (18)
- Global: mean, std, min, max, median, Q25, Q75, IQR, range, CV, skew, kurtosis
- Dwell-specific: mean, std
- Flight-specific: mean, std
- Digraph-specific: mean, std

## 📈 Models Architecture

### 1. Random Forest
- Trees: 200
- Max depth: 20
- Min samples split: 5

### 2. XGBoost
- Estimators: 200
- Max depth: 10
- Learning rate: 0.1

### 3. HGBT
- Iterations: 200
- Max depth: 10
- Learning rate: 0.1

### 4. MLP (4 layers)
- Dense(256) + Dropout(0.3)
- Dense(128) + Dropout(0.3)
- Dense(64) + Dropout(0.2)
- Dense(n_classes) softmax

### 5. 1D CNN
- Conv1D(64) + BatchNorm + MaxPool + Dropout
- Conv1D(128) + BatchNorm + MaxPool + Dropout
- Conv1D(64) + BatchNorm + GlobalAvgPool
- Dense(128) + Dropout(0.4)
- Dense(n_classes) softmax

## 🔍 Evaluation Metrics

### Identification
- **Accuracy**: % of correctly identified users
- **Confusion Matrix**: Visual of predictions
- **Precision/Recall/F1**: Per-class performance

### Verification  
- **ROC Curve**: True Positive Rate vs False Positive Rate
- **AUC**: Area under ROC (higher = better)
- **EER**: Equal Error Rate (lower = better)
- **FAR**: False Acceptance Rate
- **FRR**: False Rejection Rate

## 💡 Usage

```bash
# Run the complete pipeline
python keystroke_training_pipeline.py
```

## 🎓 What This Demonstrates

✅ Complete production-ready ML/DL pipeline  
✅ Feature engineering from raw data  
✅ Multiple model comparisons  
✅ Both classification & verification tasks  
✅ Comprehensive evaluation  
✅ Model persistence  
✅ Professional visualizations  
✅ Clean, modular, documented code  

## 📝 Notes

- Dataset: CMU Keystroke Dynamics Benchmark
- Password: `.tie5Roanl`
- 51 users, 400 samples per user
- Train/Test split: 80/20
- Random seed: 42 (reproducible)

---

**Training in progress... Check terminal for real-time updates!**
