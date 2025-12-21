# 📁 Project Structure

```
keystroke/model training/
│
├── 📄 DSL-StrongPasswordData.csv          # CMU Keystroke Dataset
├── 📄 keystroke_training_pipeline.py      # Main training script
├── 📄 organize_files.py                   # File organization script
├── 📄 requirements.txt                    # Python dependencies
│
├── 📖 README.md                           # Full documentation
├── 📖 QUICK_START.md                      # Quick reference guide
├── 📖 RESULTS_SUMMARY.md                  # Training results & analysis
│
├── 🤖 models/                             # All trained models (7 files)
│   ├── rf_model.pkl
│   ├── xgb_model.json
│   ├── hgb_model.pkl
│   ├── mlp_model.h5
│   ├── cnn_model.h5
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── 📊 confusion_matrices/                 # Confusion matrices (5 files)
│   ├── Random_Forest_confusion_matrix.png
│   ├── XGBoost_confusion_matrix.png
│   ├── HGBT_confusion_matrix.png
│   ├── MLP_confusion_matrix.png
│   └── 1D_CNN_confusion_matrix.png
│
├── 📈 feature_importance/                 # Feature importance plots (2 files)
│   ├── Random_Forest_feature_importance.png
│   └── XGBoost_feature_importance.png
│
├── 📉 training_history/                   # Neural network training curves (2 files)
│   ├── MLP_training_history.png
│   └── 1D_CNN_training_history.png
│
├── 🔐 verification/                       # Verification analysis (3 files)
│   ├── verification_roc_all_models.png
│   ├── verification_roc_curve.png
│   └── distance_distribution.png
│
└── 📋 results/                            # Final results (2 files)
    ├── model_comparison_table.csv
    └── model_comparison_charts.png
```

## 📊 Summary

- **Total Files**: 21 generated files
- **Organized into**: 6 folders
- **Models Trained**: 5
- **Visualizations**: 12 images
- **Data Tables**: 1 CSV

## 🎯 Best Model: XGBoost
- **Accuracy**: 94.53%
- **AUC**: 0.9475
- **EER**: 12.07%

See `RESULTS_SUMMARY.md` for detailed analysis!
