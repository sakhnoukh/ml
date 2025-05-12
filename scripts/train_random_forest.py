"""
Train a Random Forest model for computer price prediction.

This script trains a Random Forest model using key features and saves it for use in the app.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load the data
data_path = './data/db_computers_2025_processed.csv'
df = pd.read_csv(data_path)

# Define features and target
features = ['RAM_Número de ranuras para memoria RAM', 'Procesador_Número de núcleos del procesador']

# Convert CPU cores to numeric
df['Procesador_Número de núcleos del procesador'] = df['Procesador_Número de núcleos del procesador'].str.extract('(\\d+)').astype(float)

X = df[features]
y = df['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
train_rmse = np.sqrt(np.mean((y_train - model.predict(X_train))**2))
test_rmse = np.sqrt(np.mean((y_test - model.predict(X_test))**2))

print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

# Save the model
model_path = './models/random_forest_model.joblib'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
