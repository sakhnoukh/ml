"""
Direct Fix for Price Prediction Model

Creates a simple Random Forest model that directly predicts prices for computer systems
without using complex feature engineering that might be causing prediction issues.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib

def main():
    """Create a direct price prediction model with minimal preprocessing."""
    print("\n=== Creating Direct Price Prediction Model ===")
    
    # Load data
    data_path = './data/db_computers_2025_processed.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Show basic statistics
    print(f"Dataset shape: {df.shape}")
    print(f"Price range: €{df['Price'].min():.2f} to €{df['Price'].max():.2f}")
    print(f"Average price: €{df['Price'].mean():.2f}")
    
    # Create train/test split
    print("\nSplitting data into train/test sets...")
    X = df.drop('Price', axis=1)
    y = df['Price'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a simple model that can handle mixed data types
    print("\nTraining Random Forest model for direct price prediction...")
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
    
    # Only use numeric columns for simplicity (we can add categorical later if needed)
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"Using {len(numeric_cols)} numeric features for prediction")
    
    # Create a minimal preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols)
        ],
        remainder='drop'  # Drop other columns
    )
    
    # Create the full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Fit the pipeline
    print("Fitting the model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model performance...")
    y_pred = pipeline.predict(X_test)
    
    # Ensure predictions are always positive and reasonable
    y_pred = np.maximum(y_pred, 100)
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
    
    print(f"Test RMSE: €{rmse:.2f}")
    print(f"Test MAE: €{mae:.2f}")
    print(f"Test R²: {r2:.4f}")
    
    # Verify predictions on some test samples
    print("\nSample predictions:")
    for i in range(5):
        print(f"True: €{y_test[i]:.2f}, Predicted: €{max(100, y_pred[i]):.2f}")
    
    # Create a wrapper that ensures positive predictions
    class DirectPricePredictor:
        def __init__(self, pipeline, min_price=100):
            self.pipeline = pipeline
            self.min_price = min_price
        
        def predict(self, X):
            """Predict prices and ensure they're positive."""
            predictions = self.pipeline.predict(X)
            return np.maximum(predictions, self.min_price)
    
    # Create the final predictor
    direct_predictor = DirectPricePredictor(pipeline)
    
    # Save the model
    print("\nSaving direct price predictor...")
    os.makedirs('./models', exist_ok=True)
    joblib.dump(direct_predictor, './models/direct_price_predictor.joblib')
    
    print("Model saved to ./models/direct_price_predictor.joblib")
    print("Update app.py to use this direct predictor for realistic price predictions.")

if __name__ == "__main__":
    main()
