"""
Create Simple Price Prediction Model

This script creates a simple model that directly maps computer specifications to prices
without complex transformations that could cause prediction issues.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Add src to path
sys.path.append('./src')

# Import project modules
from features import engineer_features

def main():
    """Create and save a simple price prediction model."""
    print("\n=== Creating Simple Price Prediction Model ===")
    
    # Load data
    data_path = './data/db_computers_2025_processed.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Show basic statistics
    print(f"Dataset shape: {df.shape}")
    print(f"Price range: €{df['Price'].min():.2f} to €{df['Price'].max():.2f}")
    print(f"Average price: €{df['Price'].mean():.2f}")
    
    # Create the feature engineering pipeline
    print("\nCreating feature engineering pipeline...")
    feature_pipeline = engineer_features(df)
    
    # Extract price for training
    y = df['Price'].values
    
    # Use pipeline to transform features
    print("Transforming features...")
    X = feature_pipeline.transform(df.drop('Price', axis=1))
    
    # Train a simple RandomForest model (more robust than linear models)
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
    model.fit(X, y)
    
    # Make predictions on the training data
    print("Evaluating model...")
    y_pred = model.predict(X)
    
    # Ensure predictions are positive
    y_pred = np.maximum(y_pred, 100)
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    mae = np.mean(np.abs(y - y_pred))
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    
    print(f"Training RMSE: €{rmse:.2f}")
    print(f"Training MAE: €{mae:.2f}")
    print(f"Training R²: {r2:.4f}")
    
    # Create a simple predictor class
    class SimplePricePredictor:
        """Simple price predictor that ensures positive predictions."""
        
        def __init__(self, model, min_price=100):
            self.model = model
            self.min_price = min_price
        
        def predict(self, X):
            """Predict prices with minimum threshold."""
            predictions = self.model.predict(X)
            return np.maximum(predictions, self.min_price)
    
    # Create the predictor object
    price_predictor = SimplePricePredictor(model)
    
    # Save the model and pipeline
    print("\nSaving model and pipeline...")
    os.makedirs('./models', exist_ok=True)
    joblib.dump(feature_pipeline, './models/simple_feature_pipeline.joblib')
    joblib.dump(price_predictor, './models/simple_price_model.joblib')
    
    print("\nSimple price model and pipeline saved to models/ directory.")
    print("Update the app.py file to use these files instead.")

if __name__ == "__main__":
    main()
