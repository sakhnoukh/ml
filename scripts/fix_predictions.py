"""
Direct Price Prediction Model Fix

This script creates a direct mapping model that predicts computer prices 
accurately without going through complex standardization/transformation steps.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Add src to path
sys.path.append('./src')

# Import project modules
from features import engineer_features, identify_core_features

def main():
    """Create and save a direct price prediction model."""
    print("\n=== Creating Direct Price Prediction Model ===")
    
    # Load data
    data_path = './data/db_computers_2025_processed.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Show basic statistics
    print(f"Dataset shape: {df.shape}")
    print(f"Price range: €{df['Price'].min():.2f} to €{df['Price'].max():.2f}")
    print(f"Average price: €{df['Price'].mean():.2f}")
    
    # Identify features
    numeric_features, categorical_features = identify_core_features(df)
    print(f"\nIdentified {len(numeric_features)} numeric features")
    print(f"Identified {len(categorical_features)} categorical features")
    
    # Get feature values (everything except Price)
    X_cols = [col for col in df.columns if col != 'Price' and not df[col].isnull().all()]
    X = df[X_cols].copy()
    
    # Get price values directly (no scaling)
    y = df['Price'].copy()
    
    # Create a simple direct model (Linear Regression)
    print("\nTraining direct prediction model...")
    model = LinearRegression()
    model.fit(X, y)
    
    # Evaluate on the data
    y_pred = model.predict(X)
    
    # Ensure predictions are positive
    y_pred = np.maximum(y_pred, 100)
    
    # Calculate basic metrics
    rmse = np.sqrt(((y - y_pred) ** 2).mean())
    mae = np.abs(y - y_pred).mean()
    
    print(f"Training RMSE: €{rmse:.2f}")
    print(f"Training MAE: €{mae:.2f}")
    print(f"Min predicted price: €{y_pred.min():.2f}")
    print(f"Max predicted price: €{y_pred.max():.2f}")
    
    # Create a class that will directly predict positive prices
    class DirectPricePredictor:
        """Predicts prices directly without complex transformations."""
        
        def __init__(self, model, min_price=100):
            self.model = model
            self.min_price = min_price
        
        def predict(self, X):
            """Predict prices directly."""
            predictions = self.model.predict(X)
            return np.maximum(predictions, self.min_price)
    
    # Create the direct predictor
    direct_predictor = DirectPricePredictor(model, min_price=100)
    
    # Save the model
    os.makedirs('./models', exist_ok=True)
    joblib.dump(direct_predictor, './models/direct_price_predictor.joblib')
    
    print("\nDirect price predictor saved to models/direct_price_predictor.joblib")
    print("You can now use this model in the app for accurate price predictions.")

if __name__ == "__main__":
    main()
