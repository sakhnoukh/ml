"""
Train the final improved model with RAM capacity for more accurate price prediction.

This script extracts RAM capacity in GB and adds it as a key feature.
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

# Extract RAM capacity from 'RAM_Memoria RAM'
ram_col = 'RAM_Memoria RAM'

def extract_ram_gb(ram_str):
    if pd.isna(ram_str):
        return 8  # default value
    
    # Convert to string
    ram_str = str(ram_str)
    
    # Extract numeric value
    match = re.search(r'(\d+)', ram_str)
    if not match:
        return 8  # default if no match
    
    # Get the value
    value = float(match.group(1))
    
    return value

# Apply conversion to RAM capacity
print("Extracting RAM capacity in GB...")
df['RAM_GB'] = df[ram_col].apply(extract_ram_gb)

# Add SSD storage feature
ssd_col = 'Disco duro_Capacidad de memoria SSD'

# Function to extract numeric values from storage string
def extract_storage_gb(storage_str):
    if pd.isna(storage_str):
        return 256  # default value
    
    # Convert to string
    storage_str = str(storage_str)
    
    # Extract numeric value
    match = re.search(r'(\d[\d\.,]*)', storage_str)
    if not match:
        return 256  # default if no match
    
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
    'RAM_GB',  # NEW: Actual RAM capacity in GB
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
    "Random Forest (200 trees)": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42, learning_rate=0.1)
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
model_path = './models/final_model.joblib'
joblib.dump(pipeline, model_path)
print(f"\nModel saved to {model_path}")

# Also save feature information
feature_info = {
    'feature_list': features,
    'feature_importance': dict(zip(features, importances))
}
joblib.dump(feature_info, './models/final_model_features.joblib')
print("Feature information saved to ./models/final_model_features.joblib")

# Generate some sample predictions for common configurations
print("\nSample predictions:")
sample_configs = [
    {"name": "Basic laptop", "ram": 8, "cpu_cores": 4, "ssd": 256, "gpu": 2, "hdd": 1},
    {"name": "Mid-range laptop", "ram": 16, "cpu_cores": 6, "ssd": 512, "gpu": 4, "hdd": 1},
    {"name": "Gaming laptop", "ram": 32, "cpu_cores": 8, "ssd": 1000, "gpu": 16, "hdd": 1},
    {"name": "Workstation", "ram": 64, "cpu_cores": 16, "ssd": 2000, "gpu": 16, "hdd": 2}
]

for config in sample_configs:
    # Create a feature vector
    features = np.array([[
        config["ram"],
        config["cpu_cores"],
        config["ssd"],
        config["gpu"],
        config["hdd"]
    ]])
    
    # Scale features
    scaled_features = scaler.transform(features)
    
    # Predict price
    predicted_price = best_model.predict(scaled_features)[0]
    
    print(f"{config['name']}: €{predicted_price:.2f}")
