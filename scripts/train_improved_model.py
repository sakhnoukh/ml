"""
Train an improved Random Forest model for more realistic computer price prediction.

This script adds more features and uses more trees for better accuracy.
"""

import os
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the data
data_path = './data/db_computers_2025_processed.csv'
print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
print(f"Price range: €{df['Price'].min():.2f} to €{df['Price'].max():.2f}")
print(f"Mean price: €{df['Price'].mean():.2f}")

# Add SSD storage feature
ssd_col = 'Disco duro_Capacidad de memoria SSD'

# Function to extract numeric values from storage string
def extract_storage_gb(storage_str):
    if pd.isna(storage_str):
        return 0
    
    # Convert to string
    storage_str = str(storage_str)
    
    # Extract numeric value
    match = re.search(r'(\d[\d\.,]*)', storage_str)
    if not match:
        return 0
    
    # Get the value and convert commas to dots
    value = match.group(1).replace(',', '.')
    value = float(value)
    
    # Convert to GB if in TB
    if 'TB' in storage_str:
        value = value * 1000
    
    return value

# Apply conversion to SSD capacity
print("Extracting SSD storage capacity in GB...")
df['SSD_GB'] = df[ssd_col].apply(extract_storage_gb)

# Extract number of harddrives
hdd_count_col = 'Disco duro_Número de discos duros (instalados)'
df['HDD_Count'] = df[hdd_count_col].fillna(1).astype(float)

# Extract GPU cores from GPU description
gpu_col = 'Gráfica_GPU'
def extract_gpu_cores(gpu_str):
    if pd.isna(gpu_str):
        return 4  # default value
    
    # Convert to string
    gpu_str = str(gpu_str)
    
    # Check for common high-end indicators
    if 'RTX' in gpu_str or 'Radeon' in gpu_str:
        return 16
    elif '16-Core' in gpu_str:
        return 16
    elif '8-Core' in gpu_str:
        return 8
    elif '4-Core' in gpu_str:
        return 4
    
    # Extract numeric value if possible
    match = re.search(r'(\d+)[- ]?[Cc]ore', gpu_str)
    if match:
        return float(match.group(1))
    
    # Otherwise use a default value based on name
    if 'integrada' in gpu_str.lower() or 'integrated' in gpu_str.lower():
        return 2
    elif 'basic' in gpu_str.lower():
        return 2
    else:
        return 8  # mid-range default

print("Extracting GPU information...")
df['GPU_Cores'] = df[gpu_col].apply(extract_gpu_cores)

# Define features and target
print("Preparing features...")
features = [
    'RAM_Número de ranuras para memoria RAM',
    'Procesador_Número de núcleos del procesador',
    'SSD_GB',
    'GPU_Cores', 
    'HDD_Count'
]

# Convert CPU cores to numeric
print("Converting CPU cores to numeric...")
df['Procesador_Número de núcleos del procesador'] = df['Procesador_Número de núcleos del procesador'].str.extract(r'(\d+)').astype(float)

# Filter out rows with missing values in key features
for feature in features:
    df = df[~df[feature].isna()]

X = df[features]
y = df['Price']

# Scale the features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Try multiple algorithms and configurations
print("\nEvaluating different models...")
models = {
    "Random Forest (100 trees)": RandomForestRegressor(n_estimators=100, random_state=42),
    "Random Forest (200 trees)": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_score = float('inf')

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_rmse = np.sqrt(np.mean((y_train - train_preds)**2))
    test_rmse = np.sqrt(np.mean((y_test - test_preds)**2))
    
    print(f"\n{name}:")
    print(f"  Train RMSE: €{train_rmse:.2f}")
    print(f"  Test RMSE: €{test_rmse:.2f}")
    
    # Check if this is the best model
    if test_rmse < best_score:
        best_score = test_rmse
        best_model = model
        best_name = name

print(f"\nBest model: {best_name} with Test RMSE: €{best_score:.2f}")

# Feature importance for the best model
importances = best_model.feature_importances_
for i, feature in enumerate(features):
    print(f"Importance of {feature}: {importances[i]:.4f}")

# Create a pipeline that includes scaling
pipeline = Pipeline([
    ('scaler', scaler),
    ('model', best_model)
])

# Save the model pipeline and feature list
model_path = './models/improved_model.joblib'
joblib.dump(pipeline, model_path)
print(f"\nModel saved to {model_path}")

# Also save feature information
feature_info = {
    'feature_list': features,
    'feature_importance': dict(zip(features, importances))
}
joblib.dump(feature_info, './models/improved_model_features.joblib')
print("Feature information saved to ./models/improved_model_features.joblib")
