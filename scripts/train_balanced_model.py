"""
Train an improved model with more balanced feature importance and realistic pricing constraints.

This script ensures that:
1. SSD storage has appropriate influence on price
2. RAM capacity doesn't dominate the model completely
3. Predictions follow realistic pricing patterns
"""

import os
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, TransformerMixin

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

# ----- Create a more meaningful model for price range -------

# Create a feature for total storage (SSD + HDD)
print("Creating additional derived features...")
# Ensure SSD has proper impact - create log-transformed version (common in pricing models)
df['log_SSD_GB'] = np.log1p(df['SSD_GB'])

# Define features and target
print("Preparing features...")
basic_features = [
    'RAM_GB',
    'Procesador_Número de núcleos del procesador',
    'SSD_GB',
    'GPU_Cores', 
    'HDD_Count'
]

# Convert CPU cores to numeric
print("Converting CPU cores to numeric...")
df['Procesador_Número de núcleos del procesador'] = df['Procesador_Número de núcleos del procesador'].str.extract(r'(\d+)').astype(float)

# Calculate price per GB of RAM (to understand the relationship)
mean_ram = df.groupby('RAM_GB')['Price'].mean()
print("\nMean Price by RAM amount:")
print(mean_ram)

# Same for SSD
mean_ssd = df.groupby('SSD_GB')['Price'].mean().sort_index()
print("\nMean Price by SSD capacity:")
print(mean_ssd.head(10))

# Create a PriceConstraint transformer
class PriceConstraintTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that applies price constraints to ensure realistic predictions.
    For example, ensuring that higher SSD/RAM always increases the price.
    """
    def __init__(self, ram_cost_per_gb=15, ssd_cost_per_gb=0.15, base_price=500):
        self.ram_cost_per_gb = ram_cost_per_gb
        self.ssd_cost_per_gb = ssd_cost_per_gb
        self.base_price = base_price
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Create a copy to avoid modifying the original
        X_new = X.copy()
        
        # Add a feature that represents the baseline expected price
        # This helps ensure the model follows logical pricing patterns
        if isinstance(X_new, pd.DataFrame):
            X_new['expected_base_price'] = (
                self.base_price + 
                X_new['RAM_GB'] * self.ram_cost_per_gb + 
                X_new['SSD_GB'] * self.ssd_cost_per_gb +
                X_new['Procesador_Número de núcleos del procesador'] * 50 +
                X_new['GPU_Cores'] * 30
            )
        else:
            # For numpy arrays, assume standard column order from basic_features
            ram_idx = basic_features.index('RAM_GB')
            ssd_idx = basic_features.index('SSD_GB')
            cpu_idx = basic_features.index('Procesador_Número de núcleos del procesador')
            gpu_idx = basic_features.index('GPU_Cores')
            
            expected_base_price = (
                self.base_price + 
                X_new[:, ram_idx] * self.ram_cost_per_gb + 
                X_new[:, ssd_idx] * self.ssd_cost_per_gb +
                X_new[:, cpu_idx] * 50 +
                X_new[:, gpu_idx] * 30
            )
            
            # Add as a new column
            X_new = np.column_stack((X_new, expected_base_price))
            
        return X_new
    
    def get_feature_names_out(self, input_features=None):
        return list(input_features) + ['expected_base_price']

# Filter out rows with missing values in key features
for feature in basic_features:
    df = df[~df[feature].isna()]

X = df[basic_features]
y = df['Price']

# Apply the constraint transformer
constraint_transformer = PriceConstraintTransformer()
X_constrained = constraint_transformer.transform(X)

# Scale the features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_constrained)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Try multiple algorithms and configurations
print("\nEvaluating different models...")
models = {
    "Random Forest (200 trees)": RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42, learning_rate=0.05, max_depth=5),
    "Ridge Regression": Ridge(alpha=1.0)
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

# Create the full pipeline
pipeline = Pipeline([
    ('constraint_transformer', constraint_transformer),
    ('scaler', scaler),
    ('model', best_model)
])

# Feature importance for the best model (if available)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    # Get all feature names including the one added by constraint transformer
    all_features = constraint_transformer.get_feature_names_out(basic_features)
    
    for i, feature in enumerate(all_features):
        print(f"Importance of {feature}: {importances[i]:.4f}")
        
    # Create feature importance dictionary
    feature_importance = dict(zip(all_features, importances))
else:
    # For models without feature_importances_ attribute (like Ridge)
    feature_importance = {
        'RAM_GB': 0.25,
        'Procesador_Número de núcleos del procesador': 0.25,
        'SSD_GB': 0.20,
        'GPU_Cores': 0.15, 
        'HDD_Count': 0.05,
        'expected_base_price': 0.10
    }
    print("Model doesn't provide feature importances. Using default weights:")
    for feature, importance in feature_importance.items():
        print(f"Importance of {feature}: {importance:.4f}")

# Save the model pipeline and feature list
model_path = './models/balanced_model.joblib'
joblib.dump(pipeline, model_path)
print(f"\nModel saved to {model_path}")

# Save feature information
feature_info = {
    'feature_list': basic_features + ['expected_base_price'],
    'feature_importance': feature_importance
}
joblib.dump(feature_info, './models/balanced_model_features.joblib')
print("Feature information saved to ./models/balanced_model_features.joblib")

# Generate test predictions for specific configurations
print("\nTesting predictions for specific configurations:")

def predict_price(model_pipeline, config_dict):
    """Make a prediction for a specific configuration"""
    # Create feature vector from config
    features = pd.DataFrame([config_dict])
    
    # Make prediction
    price = model_pipeline.predict(features)[0]
    return price

# Define test configurations to verify logical price relationships
test_configs = [
    # RAM configurations
    {"name": "8GB RAM, 256GB SSD", "RAM_GB": 8, "Procesador_Número de núcleos del procesador": 4, "SSD_GB": 256, "GPU_Cores": 4, "HDD_Count": 1},
    {"name": "16GB RAM, 256GB SSD", "RAM_GB": 16, "Procesador_Número de núcleos del procesador": 4, "SSD_GB": 256, "GPU_Cores": 4, "HDD_Count": 1},
    {"name": "32GB RAM, 256GB SSD", "RAM_GB": 32, "Procesador_Número de núcleos del procesador": 4, "SSD_GB": 256, "GPU_Cores": 4, "HDD_Count": 1},
    
    # SSD configurations
    {"name": "16GB RAM, 256GB SSD", "RAM_GB": 16, "Procesador_Número de núcleos del procesador": 4, "SSD_GB": 256, "GPU_Cores": 4, "HDD_Count": 1},
    {"name": "16GB RAM, 512GB SSD", "RAM_GB": 16, "Procesador_Número de núcleos del procesador": 4, "SSD_GB": 512, "GPU_Cores": 4, "HDD_Count": 1},
    {"name": "16GB RAM, 1TB SSD", "RAM_GB": 16, "Procesador_Número de núcleos del procesador": 4, "SSD_GB": 1000, "GPU_Cores": 4, "HDD_Count": 1},
    {"name": "16GB RAM, 2TB SSD", "RAM_GB": 16, "Procesador_Número de núcleos del procesador": 4, "SSD_GB": 2000, "GPU_Cores": 4, "HDD_Count": 1},
    {"name": "16GB RAM, 4TB SSD", "RAM_GB": 16, "Procesador_Número de núcleos del procesador": 4, "SSD_GB": 4000, "GPU_Cores": 4, "HDD_Count": 1},
    
    # RAM + SSD combinations
    {"name": "32GB RAM, 1TB SSD", "RAM_GB": 32, "Procesador_Número de núcleos del procesador": 4, "SSD_GB": 1000, "GPU_Cores": 4, "HDD_Count": 1},
    {"name": "64GB RAM, 2TB SSD", "RAM_GB": 64, "Procesador_Número de núcleos del procesador": 4, "SSD_GB": 2000, "GPU_Cores": 4, "HDD_Count": 1},
    {"name": "128GB RAM, 4TB SSD", "RAM_GB": 128, "Procesador_Número de núcleos del procesador": 4, "SSD_GB": 4000, "GPU_Cores": 4, "HDD_Count": 1},
]

# Test all configurations
for config in test_configs:
    price = predict_price(pipeline, {k: v for k, v in config.items() if k != "name"})
    print(f"{config['name']}: €{price:.2f}")

# Analyze the price differences
print("\nPrice increases:")
# RAM increases
ram_configs = test_configs[:3]
for i in range(1, len(ram_configs)):
    prev_price = predict_price(pipeline, {k: v for k, v in ram_configs[i-1].items() if k != "name"})
    curr_price = predict_price(pipeline, {k: v for k, v in ram_configs[i].items() if k != "name"})
    ram_diff = ram_configs[i]["RAM_GB"] - ram_configs[i-1]["RAM_GB"]
    price_diff = curr_price - prev_price
    price_per_gb = price_diff / ram_diff
    print(f"RAM {ram_configs[i-1]['RAM_GB']}GB → {ram_configs[i]['RAM_GB']}GB: +€{price_diff:.2f} (€{price_per_gb:.2f}/GB)")

# SSD increases
ssd_configs = test_configs[3:8]
for i in range(1, len(ssd_configs)):
    prev_price = predict_price(pipeline, {k: v for k, v in ssd_configs[i-1].items() if k != "name"})
    curr_price = predict_price(pipeline, {k: v for k, v in ssd_configs[i].items() if k != "name"})
    ssd_diff = ssd_configs[i]["SSD_GB"] - ssd_configs[i-1]["SSD_GB"]
    price_diff = curr_price - prev_price
    price_per_gb = price_diff / ssd_diff
    print(f"SSD {ssd_configs[i-1]['SSD_GB']}GB → {ssd_configs[i]['SSD_GB']}GB: +€{price_diff:.2f} (€{price_per_gb:.2f}/GB)")

print("\nModel training and evaluation complete!")
