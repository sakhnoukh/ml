"""
Create a simplified direct model without custom transformers for reliable deployment.

This script creates a straightforward RandomForest model that can be loaded
without dependencies on custom classes.
"""

import os
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

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
        return 8
    
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
        return 256
    
    # Convert to string
    storage_str = str(storage_str)
    
    # Extract numeric value
    match = re.search(r'(\d[\d\.,]*)', storage_str)
    if not match:
        return 256
    
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
        return 4
    
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
        return 8

print("Extracting GPU information...")
df['GPU_Cores'] = df[gpu_col].apply(extract_gpu_cores)

# Convert CPU cores to numeric FIRST, then use for the expected price
print("Converting CPU cores to numeric...")
cpu_core_col = 'Procesador_Número de núcleos del procesador'

def extract_cpu_cores(cpu_str):
    if pd.isna(cpu_str):
        return 4.0  # default
    
    # Convert to string
    cpu_str = str(cpu_str)
    
    # Extract numeric value
    match = re.search(r'(\d+)', cpu_str)
    if match:
        return float(match.group(1))
    else:
        return 4.0  # default if no match

df['CPU_Cores'] = df[cpu_core_col].apply(extract_cpu_cores)

# Create a feature for the expected price based on component costs
print("Creating expected price feature...")
df['expected_price'] = (
    500 +  # Base price
    df['RAM_GB'] * 15 +  # RAM cost
    df['SSD_GB'] * 0.15 +  # SSD cost
    df['CPU_Cores'] * 50 +  # CPU cost
    df['GPU_Cores'] * 30  # GPU cost
)

# Define features and target
print("Preparing features...")
features = [
    'RAM_GB',
    'CPU_Cores',
    'SSD_GB',
    'GPU_Cores', 
    'HDD_Count',
    'expected_price'
]

# Filter out rows with missing values in key features
for feature in features:
    df = df[~df[feature].isna()]

X = df[features]
y = df['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest model
print("Training Random Forest model...")
model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# Evaluate the model
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_rmse = np.sqrt(np.mean((y_train - train_preds)**2))
test_rmse = np.sqrt(np.mean((y_test - test_preds)**2))

print(f"Train RMSE: €{train_rmse:.2f}")
print(f"Test RMSE: €{test_rmse:.2f}")

# Feature importance
importances = model.feature_importances_
for i, feature in enumerate(features):
    print(f"Importance of {feature}: {importances[i]:.4f}")

# Save the model
model_path = './models/direct_model.joblib'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Save feature information
feature_info = {
    'feature_list': features,
    'feature_importance': dict(zip(features, importances))
}
joblib.dump(feature_info, './models/direct_model_features.joblib')
print("Feature information saved to ./models/direct_model_features.joblib")

# Test the model on typical configurations
print("\nTesting model with typical configurations:")
test_configs = [
    {"name": "8GB RAM, 256GB SSD", "ram": 8, "cpu": 4, "ssd": 256, "gpu": 4, "hdd": 1},
    {"name": "16GB RAM, 256GB SSD", "ram": 16, "cpu": 4, "ssd": 256, "gpu": 4, "hdd": 1},
    {"name": "16GB RAM, 512GB SSD", "ram": 16, "cpu": 4, "ssd": 512, "gpu": 4, "hdd": 1},
    {"name": "16GB RAM, 1TB SSD", "ram": 16, "cpu": 4, "ssd": 1000, "gpu": 4, "hdd": 1},
    {"name": "32GB RAM, 1TB SSD", "ram": 32, "cpu": 6, "ssd": 1000, "gpu": 8, "hdd": 1},
    {"name": "64GB RAM, 2TB SSD", "ram": 64, "cpu": 8, "ssd": 2000, "gpu": 16, "hdd": 1},
    {"name": "128GB RAM, 4TB SSD", "ram": 128, "cpu": 16, "ssd": 4000, "gpu": 16, "hdd": 2}
]

for config in test_configs:
    # Create a DataFrame for the test configuration
    test_df = pd.DataFrame({
        'RAM_GB': [config["ram"]],
        'CPU_Cores': [config["cpu"]],
        'SSD_GB': [config["ssd"]],
        'GPU_Cores': [config["gpu"]],
        'HDD_Count': [config["hdd"]],
        'expected_price': [500 + config["ram"]*15 + config["ssd"]*0.15 + config["cpu"]*50 + config["gpu"]*30]
    })
    
    # Predict price
    predicted_price = model.predict(test_df)[0]
    
    print(f"{config['name']}: €{predicted_price:.2f}")
