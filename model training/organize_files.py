"""
Organize generated files into folders
"""
import os
import shutil

# Create folders
folders = {
    'models': ['rf_model.pkl', 'xgb_model.json', 'hgb_model.pkl', 
               'mlp_model.h5', 'cnn_model.h5', 'scaler.pkl', 'label_encoder.pkl'],
    'confusion_matrices': ['Random_Forest_confusion_matrix.png', 
                          'XGBoost_confusion_matrix.png',
                          'HGBT_confusion_matrix.png',
                          'MLP_confusion_matrix.png',
                          '1D_CNN_confusion_matrix.png'],
    'feature_importance': ['Random_Forest_feature_importance.png',
                          'XGBoost_feature_importance.png'],
    'training_history': ['MLP_training_history.png',
                        '1D_CNN_training_history.png'],
    'verification': ['verification_roc_curve.png',
                    'verification_roc_all_models.png',
                    'distance_distribution.png'],
    'results': ['model_comparison_table.csv',
               'model_comparison_charts.png']
}

# Create directories
for folder in folders.keys():
    os.makedirs(folder, exist_ok=True)
    print(f"✓ Created folder: {folder}/")

# Move files
moved_count = 0
for folder, files in folders.items():
    for file in files:
        if os.path.exists(file):
            shutil.move(file, os.path.join(folder, file))
            print(f"  Moved: {file} → {folder}/")
            moved_count += 1

print(f"\n✓ Organized {moved_count} files into {len(folders)} folders!")
print("\nFolder structure:")
for folder in folders.keys():
    files_in_folder = [f for f in folders[folder] if os.path.exists(os.path.join(folder, f))]
    print(f"  {folder}/ ({len(files_in_folder)} files)")
