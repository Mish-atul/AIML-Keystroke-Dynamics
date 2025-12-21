"""
Keystroke Dynamics Authentication Pipeline
Complete ML/DL training system for user identification and verification
using the CMU Keystroke Dynamics Benchmark Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc, roc_auc_score)
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import pickle
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class KeystrokeDynamicsTrainer:
    """Complete training pipeline for keystroke dynamics authentication"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.features = None
        self.labels = None
        self.sequences = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        self.verification_results = {}
        
    def load_dataset(self):
        """Load the CMU keystroke dynamics dataset"""
        print("=" * 80)
        print("LOADING DATASET")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Number of subjects: {self.df['subject'].nunique()}")
        print(f"Columns: {len(self.df.columns)}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        print(f"\nDataset info:")
        print(self.df.info())
        
        return self.df
    
    def extract_features(self):
        """
        Extract comprehensive features from raw keystroke data:
        1. Dwell times (hold times)
        2. Flight times (up-down times)
        3. Digraph latencies (down-down times)
        4. Statistical features (mean, std, min, max, median, etc.)
        """
        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING")
        print("=" * 80)
        
        # Get column names
        all_cols = self.df.columns.tolist()
        
        # Extract different feature types
        dwell_cols = [col for col in all_cols if col.startswith('H.')]
        flight_cols = [col for col in all_cols if col.startswith('UD.')]
        digraph_cols = [col for col in all_cols if col.startswith('DD.')]
        
        print(f"Dwell time features: {len(dwell_cols)}")
        print(f"Flight time features: {len(flight_cols)}")
        print(f"Digraph latency features: {len(digraph_cols)}")
        
        # Combine all timing features
        timing_cols = dwell_cols + flight_cols + digraph_cols
        
        # Extract raw timing features
        feature_data = self.df[timing_cols].values
        
        # Calculate statistical features for each sample
        print("\nCalculating statistical features...")
        statistical_features = []
        
        # Vectorized statistical calculations for speed
        feature_data_t = feature_data.T  # Transpose for easier column operations
        
        # Global statistics across all timing features
        means = np.mean(feature_data, axis=1)
        stds = np.std(feature_data, axis=1)
        mins = np.min(feature_data, axis=1)
        maxs = np.max(feature_data, axis=1)
        medians = np.median(feature_data, axis=1)
        q25s = np.percentile(feature_data, 25, axis=1)
        q75s = np.percentile(feature_data, 75, axis=1)
        iqrs = q75s - q25s
        ranges = maxs - mins
        cvs = stds / (means + 1e-10)
        
        # Feature-specific statistics
        dwell_data = feature_data[:, :len(dwell_cols)]
        dwell_means = np.mean(dwell_data, axis=1)
        dwell_stds = np.std(dwell_data, axis=1)
        
        flight_start = len(dwell_cols)
        flight_end = flight_start + len(flight_cols)
        flight_data = feature_data[:, flight_start:flight_end]
        flight_means = np.mean(flight_data, axis=1)
        flight_stds = np.std(flight_data, axis=1)
        
        digraph_start = len(dwell_cols) + len(flight_cols)
        digraph_data = feature_data[:, digraph_start:]
        digraph_means = np.mean(digraph_data, axis=1)
        digraph_stds = np.std(digraph_data, axis=1)
        
        # Stack all statistical features
        statistical_features = np.column_stack([
            means, stds, mins, maxs, medians, q25s, q75s, iqrs, ranges, cvs,
            dwell_means, dwell_stds, flight_means, flight_stds, digraph_means, digraph_stds
        ])
        
        print(f"Statistical features computed: {statistical_features.shape[1]} features")
        
        # Combine raw features with statistical features
        self.features = np.hstack([feature_data, statistical_features])
        
        print(f"\nFinal feature matrix shape: {self.features.shape}")
        print(f"  - Raw timing features: {feature_data.shape[1]}")
        print(f"  - Statistical features: {statistical_features.shape[1]}")
        print(f"  - Total features: {self.features.shape[1]}")
        
        # Extract labels
        self.labels = self.df['subject'].values
        print(f"\nLabels shape: {self.labels.shape}")
        print(f"Unique users: {len(np.unique(self.labels))}")
        
        return self.features, self.labels
    
    def prepare_sequences(self):
        """Prepare sequence data for CNN model"""
        print("\n" + "=" * 80)
        print("PREPARING SEQUENCES FOR CNN")
        print("=" * 80)
        
        # Get timing columns only (not statistical features)
        all_cols = self.df.columns.tolist()
        timing_cols = [col for col in all_cols if col.startswith(('H.', 'UD.', 'DD.'))]
        
        # Reshape for CNN: (samples, timesteps, features)
        # We'll treat each timing feature as a timestep
        sequence_data = self.df[timing_cols].values
        
        # Reshape to (samples, timesteps, 1) for 1D CNN
        self.sequences = sequence_data.reshape(sequence_data.shape[0], sequence_data.shape[1], 1)
        
        print(f"Sequence shape for CNN: {self.sequences.shape}")
        print(f"  - Samples: {self.sequences.shape[0]}")
        print(f"  - Timesteps: {self.sequences.shape[1]}")
        print(f"  - Features per timestep: {self.sequences.shape[2]}")
        
        return self.sequences
    
    def train_models(self):
        """Train 5 different models for user identification"""
        print("\n" + "=" * 80)
        print("TRAINING MODELS FOR USER IDENTIFICATION")
        print("=" * 80)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(self.labels)
        n_classes = len(np.unique(y_encoded))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Number of classes: {n_classes}")
        
        # Store train/test splits for later use
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        # 1. RANDOM FOREST
        print("\n" + "-" * 80)
        print("1. Training Random Forest Classifier")
        print("-" * 80)
        start_time = time.time()
        
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_time = time.time() - start_time
        
        rf_pred = rf_model.predict(X_test_scaled)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        self.models['Random Forest'] = rf_model
        self.results['Random Forest'] = {
            'model': rf_model,
            'predictions': rf_pred,
            'accuracy': rf_accuracy,
            'training_time': rf_time,
            'feature_importance': rf_model.feature_importances_
        }
        
        print(f"Accuracy: {rf_accuracy:.4f}")
        print(f"Training time: {rf_time:.2f}s")
        
        # 2. XGBOOST
        print("\n" + "-" * 80)
        print("2. Training XGBoost Classifier")
        print("-" * 80)
        start_time = time.time()
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        xgb_model.fit(X_train_scaled, y_train)
        xgb_time = time.time() - start_time
        
        xgb_pred = xgb_model.predict(X_test_scaled)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        
        self.models['XGBoost'] = xgb_model
        self.results['XGBoost'] = {
            'model': xgb_model,
            'predictions': xgb_pred,
            'accuracy': xgb_accuracy,
            'training_time': xgb_time,
            'feature_importance': xgb_model.feature_importances_
        }
        
        print(f"Accuracy: {xgb_accuracy:.4f}")
        print(f"Training time: {xgb_time:.2f}s")
        
        # 3. HISTOGRAM GRADIENT BOOSTING
        print("\n" + "-" * 80)
        print("3. Training Histogram Gradient Boosting Classifier")
        print("-" * 80)
        start_time = time.time()
        
        hgb_model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42
        )
        hgb_model.fit(X_train_scaled, y_train)
        hgb_time = time.time() - start_time
        
        hgb_pred = hgb_model.predict(X_test_scaled)
        hgb_accuracy = accuracy_score(y_test, hgb_pred)
        
        self.models['HGBT'] = hgb_model
        self.results['HGBT'] = {
            'model': hgb_model,
            'predictions': hgb_pred,
            'accuracy': hgb_accuracy,
            'training_time': hgb_time,
            'feature_importance': None  # HGBT doesn't provide feature importance easily
        }
        
        print(f"Accuracy: {hgb_accuracy:.4f}")
        print(f"Training time: {hgb_time:.2f}s")
        
        # 4. MLP / ANN
        print("\n" + "-" * 80)
        print("4. Training Multi-Layer Perceptron (ANN)")
        print("-" * 80)
        start_time = time.time()
        
        # Convert labels to categorical for neural network
        y_train_cat = to_categorical(y_train, num_classes=n_classes)
        y_test_cat = to_categorical(y_test, num_classes=n_classes)
        
        mlp_model = models.Sequential([
            layers.Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(n_classes, activation='softmax')
        ])
        
        mlp_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = mlp_model.fit(
            X_train_scaled, y_train_cat,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        mlp_time = time.time() - start_time
        
        mlp_pred_proba = mlp_model.predict(X_test_scaled, verbose=0)
        mlp_pred = np.argmax(mlp_pred_proba, axis=1)
        mlp_accuracy = accuracy_score(y_test, mlp_pred)
        
        self.models['MLP'] = mlp_model
        self.results['MLP'] = {
            'model': mlp_model,
            'predictions': mlp_pred,
            'accuracy': mlp_accuracy,
            'training_time': mlp_time,
            'history': history,
            'feature_importance': None
        }
        
        print(f"Accuracy: {mlp_accuracy:.4f}")
        print(f"Training time: {mlp_time:.2f}s")
        
        # 5. 1D CNN
        print("\n" + "-" * 80)
        print("5. Training 1D Convolutional Neural Network")
        print("-" * 80)
        start_time = time.time()
        
        # Prepare sequence data for CNN
        X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
            self.sequences, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Normalize sequences
        X_train_seq_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
        X_test_seq_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)
        
        seq_scaler = StandardScaler()
        X_train_seq_scaled = seq_scaler.fit_transform(X_train_seq_flat)
        X_test_seq_scaled = seq_scaler.transform(X_test_seq_flat)
        
        X_train_seq_scaled = X_train_seq_scaled.reshape(X_train_seq.shape)
        X_test_seq_scaled = X_test_seq_scaled.reshape(X_test_seq.shape)
        
        y_train_seq_cat = to_categorical(y_train_seq, num_classes=n_classes)
        y_test_seq_cat = to_categorical(y_test_seq, num_classes=n_classes)
        
        cnn_model = models.Sequential([
            layers.Conv1D(64, kernel_size=3, activation='relu', 
                         input_shape=(X_train_seq_scaled.shape[1], X_train_seq_scaled.shape[2])),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.Conv1D(128, kernel_size=3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.Conv1D(64, kernel_size=3, activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(n_classes, activation='softmax')
        ])
        
        cnn_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        cnn_history = cnn_model.fit(
            X_train_seq_scaled, y_train_seq_cat,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        cnn_time = time.time() - start_time
        
        cnn_pred_proba = cnn_model.predict(X_test_seq_scaled, verbose=0)
        cnn_pred = np.argmax(cnn_pred_proba, axis=1)
        cnn_accuracy = accuracy_score(y_test_seq, cnn_pred)
        
        self.models['1D CNN'] = cnn_model
        self.results['1D CNN'] = {
            'model': cnn_model,
            'predictions': cnn_pred,
            'accuracy': cnn_accuracy,
            'training_time': cnn_time,
            'history': cnn_history,
            'feature_importance': None,
            'y_test': y_test_seq  # Store for confusion matrix
        }
        
        print(f"Accuracy: {cnn_accuracy:.4f}")
        print(f"Training time: {cnn_time:.2f}s")
        
        print("\n" + "=" * 80)
        print("MODEL TRAINING COMPLETED")
        print("=" * 80)
        
        return self.models, self.results
    
    def evaluate_identification(self):
        """Evaluate all models for identification task"""
        print("\n" + "=" * 80)
        print("IDENTIFICATION TASK EVALUATION")
        print("=" * 80)
        
        for model_name, result in self.results.items():
            print("\n" + "=" * 80)
            print(f"MODEL: {model_name}")
            print("=" * 80)
            
            predictions = result['predictions']
            
            # Use appropriate y_test
            if model_name == '1D CNN':
                y_true = result['y_test']
            else:
                y_true = self.y_test
            
            # Accuracy
            accuracy = result['accuracy']
            print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Classification Report
            print("\nClassification Report:")
            print(classification_report(y_true, predictions, zero_division=0))
            
            # Confusion Matrix
            cm = confusion_matrix(y_true, predictions)
            
            # Plot confusion matrix
            plt.figure(figsize=(12, 10))
            
            # Limit display for large matrices
            if len(cm) > 20:
                # Show only first 20x20 for visualization
                cm_display = cm[:20, :20]
                sns.heatmap(cm_display, annot=True, fmt='d', cmap='Blues', 
                           cbar_kws={'label': 'Count'})
                plt.title(f'{model_name} - Confusion Matrix (First 20 classes)')
            else:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           cbar_kws={'label': 'Count'})
                plt.title(f'{model_name} - Confusion Matrix')
            
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f'{model_name.replace(" ", "_").replace("/", "_")}_confusion_matrix.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"\nConfusion matrix saved as '{model_name.replace(' ', '_').replace('/', '_')}_confusion_matrix.png'")
            
            # Feature Importance (if available)
            if result['feature_importance'] is not None:
                feature_imp = result['feature_importance']
                
                # Get top 20 features
                top_indices = np.argsort(feature_imp)[-20:]
                top_importance = feature_imp[top_indices]
                
                plt.figure(figsize=(10, 8))
                plt.barh(range(len(top_indices)), top_importance)
                plt.yticks(range(len(top_indices)), 
                          [f'Feature {i}' for i in top_indices])
                plt.xlabel('Importance')
                plt.title(f'{model_name} - Top 20 Feature Importance')
                plt.tight_layout()
                plt.savefig(f'{model_name.replace(" ", "_")}_feature_importance.png', 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Feature importance plot saved as '{model_name.replace(' ', '_')}_feature_importance.png'")
            
            # Training history for neural networks
            if 'history' in result and result['history'] is not None:
                history = result['history']
                
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 2, 1)
                plt.plot(history.history['accuracy'], label='Train')
                plt.plot(history.history['val_accuracy'], label='Validation')
                plt.title(f'{model_name} - Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(1, 2, 2)
                plt.plot(history.history['loss'], label='Train')
                plt.plot(history.history['val_loss'], label='Validation')
                plt.title(f'{model_name} - Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(f'{model_name.replace(" ", "_").replace("/", "_")}_training_history.png', 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Training history saved as '{model_name.replace(' ', '_').replace('/', '_')}_training_history.png'")
    
    def evaluate_verification(self):
        """Evaluate models for verification task using model-specific predictions"""
        print("\n" + "=" * 80)
        print("VERIFICATION TASK EVALUATION (MODEL-SPECIFIC)")
        print("=" * 80)
        
        # For each model, calculate verification metrics using prediction confidence
        for model_name, result in self.results.items():
            print(f"\n{'-'*80}")
            print(f"Verification for: {model_name}")
            print(f"{'-'*80}")
            
            model = result['model']
            
            # Get test data
            if model_name == '1D CNN':
                # CNN uses different test set - need to re-split to get correct test set
                _, X_test_seq, _, y_test_seq = train_test_split(
                    self.sequences, self.label_encoder.transform(self.labels), 
                    test_size=0.2, random_state=42, 
                    stratify=self.label_encoder.transform(self.labels)
                )
                
                # Normalize sequences
                X_train_seq, _, _, _ = train_test_split(
                    self.sequences, self.label_encoder.transform(self.labels),
                    test_size=0.2, random_state=42,
                    stratify=self.label_encoder.transform(self.labels)
                )
                X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
                seq_scaler = StandardScaler()
                seq_scaler.fit(X_train_flat)
                
                X_test_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)
                X_test_scaled = seq_scaler.transform(X_test_flat).reshape(X_test_seq.shape)
                
                # Get prediction probabilities
                pred_proba = model.predict(X_test_scaled, verbose=0)
                y_true = y_test_seq
            else:
                y_true = self.y_test
                X_test = self.X_test
                
                # Get prediction probabilities
                if model_name == 'MLP':
                    pred_proba = model.predict(X_test, verbose=0)
                else:
                    pred_proba = model.predict_proba(X_test)
            
            # Calculate verification metrics using max confidence score
            # For each sample, use the maximum class probability as the authentication score
            max_proba = np.max(pred_proba, axis=1)
            predictions = np.argmax(pred_proba, axis=1)
            
            # Create binary labels: correct prediction = genuine (1), wrong = impostor (0)
            is_genuine = (predictions == y_true).astype(int)
            
            # Calculate ROC
            fpr, tpr, thresholds = roc_curve(is_genuine, max_proba)
            roc_auc = auc(fpr, tpr)
            
            # Calculate EER
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
            eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
            
            # Store results for this model
            self.results[model_name]['auc'] = roc_auc
            self.results[model_name]['eer'] = eer
            self.results[model_name]['fpr'] = fpr
            self.results[model_name]['tpr'] = tpr
            
            print(f"AUC: {roc_auc:.4f}")
            print(f"EER: {eer:.4f} ({eer*100:.2f}%)")
            print(f"FAR at EER: {fpr[eer_idx]:.4f}")
            print(f"FRR at EER: {fnr[eer_idx]:.4f}")
        
        # Plot combined ROC curves for all models
        print(f"\n{'='*80}")
        print("Generating combined ROC curve plot...")
        print(f"{'='*80}")
        
        plt.figure(figsize=(12, 8))
        colors = ['darkorange', 'green', 'blue', 'red', 'purple']
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            if 'fpr' in result and 'tpr' in result:
                plt.plot(result['fpr'], result['tpr'], 
                        color=colors[idx % len(colors)], lw=2,
                        label=f'{model_name} (AUC={result["auc"]:.4f}, EER={result["eer"]:.4f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Acceptance Rate (FAR)', fontsize=12)
        plt.ylabel('True Positive Rate (1 - FRR)', fontsize=12)
        plt.title('ROC Curves - All Models (Verification Task)', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('verification_roc_all_models.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Combined ROC curve saved as 'verification_roc_all_models.png'")
        
        return self.results
    
    def compare_models(self):
        """Generate comparison table for all models"""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON TABLE")
        print("=" * 80)
        
        comparison_data = []
        
        for model_name, result in self.results.items():
            row = {
                'Model': model_name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Accuracy %': f"{result['accuracy']*100:.2f}%",
                'Training Time (s)': f"{result['training_time']:.2f}",
                'AUC': f"{result.get('auc', 0.0):.4f}",
                'EER': f"{result.get('eer', 0.0):.4f}",
                'Notes': ''
            }
            
            # Add model-specific notes
            if model_name == 'Random Forest':
                row['Notes'] = 'Ensemble method, interpretable'
            elif model_name == 'XGBoost':
                row['Notes'] = 'Gradient boosting, high performance'
            elif model_name == 'HGBT':
                row['Notes'] = 'Fast histogram-based boosting'
            elif model_name == 'MLP':
                row['Notes'] = 'Deep neural network, 4 layers'
            elif model_name == '1D CNN':
                row['Notes'] = 'Convolutional NN, sequence-based'
            
            comparison_data.append(row)
        
        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display table
        print("\n")
        print(comparison_df.to_string(index=False))
        
        # Save to CSV
        comparison_df.to_csv('model_comparison_table.csv', index=False)
        print(f"\n✓ Comparison table saved as 'model_comparison_table.csv'")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        axes[0, 0].bar(comparison_df['Model'], 
                       comparison_df['Accuracy'].astype(float),
                       color='steelblue')
        axes[0, 0].set_ylabel('Accuracy', fontsize=11)
        axes[0, 0].set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Training time comparison
        axes[0, 1].bar(comparison_df['Model'], 
                       comparison_df['Training Time (s)'].astype(float),
                       color='coral')
        axes[0, 1].set_ylabel('Time (seconds)', fontsize=11)
        axes[0, 1].set_title('Training Time Comparison', fontsize=12, fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # AUC comparison
        axes[1, 0].bar(comparison_df['Model'], 
                       comparison_df['AUC'].astype(float),
                       color='mediumseagreen')
        axes[1, 0].set_ylabel('AUC', fontsize=11)
        axes[1, 0].set_title('Verification AUC by Model', fontsize=12, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].set_ylim([0, 1])
        
        # EER comparison
        axes[1, 1].bar(comparison_df['Model'], 
                       comparison_df['EER'].astype(float),
                       color='orchid')
        axes[1, 1].set_ylabel('EER (lower is better)', fontsize=11)
        axes[1, 1].set_title('Equal Error Rate by Model', fontsize=12, fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('model_comparison_charts.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparison charts saved as 'model_comparison_charts.png'")
        
        return comparison_df
    
    def save_models(self):
        """Save all trained models to files"""
        print("\n" + "=" * 80)
        print("SAVING TRAINED MODELS")
        print("=" * 80)
        
        # 1. Save Random Forest
        with open('rf_model.pkl', 'wb') as f:
            pickle.dump(self.models['Random Forest'], f)
        print("✓ Random Forest saved as 'rf_model.pkl'")
        
        # 2. Save XGBoost
        self.models['XGBoost'].save_model('xgb_model.json')
        print("✓ XGBoost saved as 'xgb_model.json'")
        
        # 3. Save HGBT
        with open('hgb_model.pkl', 'wb') as f:
            pickle.dump(self.models['HGBT'], f)
        print("✓ Histogram Gradient Boosting saved as 'hgb_model.pkl'")
        
        # 4. Save MLP
        self.models['MLP'].save('mlp_model.h5')
        print("✓ MLP/ANN saved as 'mlp_model.h5'")
        
        # 5. Save CNN
        self.models['1D CNN'].save('cnn_model.h5')
        print("✓ 1D CNN saved as 'cnn_model.h5'")
        
        # Save scaler and label encoder
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print("✓ Feature scaler saved as 'scaler.pkl'")
        
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print("✓ Label encoder saved as 'label_encoder.pkl'")
        
        print("\n" + "=" * 80)
        print("ALL MODELS SAVED SUCCESSFULLY")
        print("=" * 80)


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("KEYSTROKE DYNAMICS AUTHENTICATION PIPELINE")
    print("CMU Benchmark Dataset")
    print("=" * 80)
    
    # Initialize trainer
    data_path = 'DSL-StrongPasswordData.csv'
    trainer = KeystrokeDynamicsTrainer(data_path)
    
    # 1. Load dataset
    trainer.load_dataset()
    
    # 2. Feature engineering
    trainer.extract_features()
    
    # 3. Prepare sequences for CNN
    trainer.prepare_sequences()
    
    # 4. Train all models
    trainer.train_models()
    
    # 5. Evaluate identification
    trainer.evaluate_identification()
    
    # 6. Evaluate verification
    trainer.evaluate_verification()
    
    # 7. Compare models
    trainer.compare_models()
    
    # 8. Save models
    trainer.save_models()
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated Files:")
    print("  Models:")
    print("    - rf_model.pkl")
    print("    - xgb_model.json")
    print("    - hgb_model.pkl")
    print("    - mlp_model.h5")
    print("    - cnn_model.h5")
    print("    - scaler.pkl")
    print("    - label_encoder.pkl")
    print("\n  Visualizations:")
    print("    - *_confusion_matrix.png (5 files)")
    print("    - *_feature_importance.png (2 files)")
    print("    - *_training_history.png (2 files)")
    print("    - verification_roc_curve.png")
    print("    - distance_distribution.png")
    print("    - model_comparison_charts.png")
    print("\n  Tables:")
    print("    - model_comparison_table.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
