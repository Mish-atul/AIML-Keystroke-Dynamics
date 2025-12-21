# Keystroke Dynamics Authentication Pipeline

Complete ML/DL training system for keystroke dynamics-based user authentication using the CMU Keystroke Dynamics Benchmark Dataset.

## Features

### 1. **Feature Engineering**
- **Dwell Time (Hold Time)**: Time a key is held down (H.x features)
- **Flight Time**: Time between releasing one key and pressing the next (UD.x.y features)
- **Digraph Latency**: Time between pressing two consecutive keys (DD.x.y features)
- **Statistical Features**: mean, std, min, max, median, quartiles, IQR, range, CV, skewness, kurtosis
- **Feature-specific statistics**: separate stats for dwell, flight, and digraph features

### 2. **Models Trained** (5 models)

#### Traditional ML Models:
1. **Random Forest Classifier** (200 trees)
2. **XGBoost Classifier** (Gradient Boosting)
3. **Histogram Gradient Boosting Classifier** (HGBT)

#### Deep Learning Models:
4. **Multi-Layer Perceptron (MLP/ANN)** - 4-layer neural network
5. **1D Convolutional Neural Network (CNN)** - Sequence-based learning

### 3. **Evaluation Metrics**

#### Identification Task:
- Accuracy
- Confusion Matrix
- Classification Report (precision, recall, F1-score)
- Feature Importance (for tree-based models)
- Training Time

#### Verification Task:
- ROC Curve
- AUC (Area Under Curve)
- FAR (False Acceptance Rate)
- FRR (False Rejection Rate)
- EER (Equal Error Rate)
- Distance distributions (genuine vs impostor)

### 4. **Outputs Generated**

#### Saved Models:
- `rf_model.pkl` - Random Forest
- `xgb_model.json` - XGBoost
- `hgb_model.pkl` - Histogram Gradient Boosting
- `mlp_model.h5` - Multi-Layer Perceptron
- `cnn_model.h5` - 1D CNN
- `scaler.pkl` - Feature scaler
- `label_encoder.pkl` - Label encoder

#### Visualizations:
- Confusion matrices for all 5 models
- Feature importance plots (RF, XGBoost)
- Training history plots (MLP, CNN)
- ROC curve for verification
- Distance distribution plot
- Model comparison charts

#### Tables:
- `model_comparison_table.csv` - Comprehensive comparison

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Simply run the main script:

```bash
python keystroke_training_pipeline.py
```

The script will:
1. Load the CMU dataset
2. Extract and engineer features
3. Train all 5 models
4. Evaluate identification performance
5. Evaluate verification performance
6. Generate comparison table
7. Save all models and visualizations

## Dataset Format

The CMU Keystroke Dynamics Benchmark Dataset contains:
- **subject**: User identifier
- **sessionIndex**: Session number
- **rep**: Repetition number
- **H.x**: Hold time for key 'x'
- **DD.x.y**: Down-down time between keys 'x' and 'y'
- **UD.x.y**: Up-down time between keys 'x' and 'y'

Password typed: `.tie5Roanl`

## Results

All results including:
- Model accuracies
- Confusion matrices
- ROC curves
- EER values
- Training times

are printed to console and saved as files in the working directory.

## Architecture Details

### Random Forest
- n_estimators: 200
- max_depth: 20
- min_samples_split: 5
- min_samples_leaf: 2

### XGBoost
- n_estimators: 200
- max_depth: 10
- learning_rate: 0.1
- subsample: 0.8

### HGBT
- max_iter: 200
- max_depth: 10
- learning_rate: 0.1

### MLP
- Layer 1: 256 neurons (ReLU) + Dropout(0.3)
- Layer 2: 128 neurons (ReLU) + Dropout(0.3)
- Layer 3: 64 neurons (ReLU) + Dropout(0.2)
- Output: n_classes (Softmax)
- Optimizer: Adam (lr=0.001)
- Epochs: 50

### 1D CNN
- Conv1D: 64 filters, kernel=3 + BatchNorm + MaxPool + Dropout(0.3)
- Conv1D: 128 filters, kernel=3 + BatchNorm + MaxPool + Dropout(0.3)
- Conv1D: 64 filters, kernel=3 + BatchNorm + GlobalAvgPool
- Dense: 128 neurons + Dropout(0.4)
- Output: n_classes (Softmax)
- Optimizer: Adam (lr=0.001)
- Epochs: 50

## Code Structure

```python
class KeystrokeDynamicsTrainer:
    - load_dataset()           # Load CSV data
    - extract_features()       # Feature engineering
    - prepare_sequences()      # Prepare data for CNN
    - train_models()           # Train all 5 models
    - evaluate_identification() # Evaluate classification
    - evaluate_verification()  # Evaluate verification
    - compare_models()         # Generate comparison table
    - save_models()            # Save all models
```

## Notes

- Training may take 5-15 minutes depending on your hardware
- GPU acceleration is used if TensorFlow detects a GPU
- All random seeds are set to 42 for reproducibility
- Large confusion matrices (>20 classes) are truncated in visualization

## Citation

Dataset: CMU Keystroke Dynamics Benchmark Dataset
Password: `.tie5Roanl`

## License

This code is provided for educational and research purposes.
