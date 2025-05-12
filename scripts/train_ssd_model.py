"""
Train a Random Forest model that includes SSD storage as a feature.

This script creates an improved model for computer price prediction.
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

# Add SSD storage feature
# We'll extract numeric values from 'Disco duro_Capacidad de memoria SSD'
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

# Define features and target
print("Preparing features...")
features = [
    'RAM_Número de ranuras para memoria RAM',
    'Procesador_Número de núcleos del procesador',
    'SSD_GB'
]

# Convert CPU cores to numeric
print("Converting CPU cores to numeric...")
df['Procesador_Número de núcleos del procesador'] = df['Procesador_Número de núcleos del procesador'].str.extract(r'(\d+)').astype(float)

X = df[features]
y = df['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("\nTraining Random Forest model with SSD feature...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
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
model_path = './models/ssd_rf_model.joblib'
joblib.dump(model, model_path)
print(f"\nModel saved to {model_path}")

# Save feature importance for the app
feature_importance = dict(zip(features, importances))
joblib.dump(feature_importance, './models/feature_importance.joblib')
print("Feature importances saved to ./models/feature_importance.joblib")
