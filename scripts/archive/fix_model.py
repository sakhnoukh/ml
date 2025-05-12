"""
Fix Price Prediction Model

Creates a new Random Forest model that directly predicts prices for the computer systems
without complex transformations, solving the issue with unrealistic predictions.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Add src to path
sys.path.append('./src')

# Import project modules
from features import identify_core_features
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def main():
    """Create and save an improved price prediction model."""
    print("\n=== Creating Fixed Price Prediction Model ===")
    
    # Load data
    data_path = './data/db_computers_2025_processed.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Show basic statistics
    print(f"Dataset shape: {df.shape}")
    print(f"Price range: €{df['Price'].min():.2f} to €{df['Price'].max():.2f}")
    print(f"Average price: €{df['Price'].mean():.2f}")
    
    # Identify numeric and categorical features
    numeric_features, categorical_features = identify_core_features(df)
    print(f"\nIdentified {len(numeric_features)} numeric features")
    print(f"Identified {len(categorical_features)} categorical features")
    
    # Create train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training set: {train_df.shape[0]} samples")
    print(f"Test set: {test_df.shape[0]} samples")
    
    # Create a simple preprocessing pipeline
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create the column transformer
    feature_columns = numeric_features + categorical_features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    # Create the full pipeline with a RandomForest model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42))
    ])
    
    # Train the pipeline
    print("\nTraining Random Forest model...")
    X_train = train_df.drop('Price', axis=1)
    y_train = train_df['Price'].values
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    X_test = test_df.drop('Price', axis=1)
    y_test = test_df['Price'].values
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
    
    print(f"Test RMSE: €{rmse:.2f}")
    print(f"Test MAE: €{mae:.2f}")
    print(f"Test R²: {r2:.4f}")
    
    # Create the final model (ensuring predictions are always positive)
    class PositivePriceModel:
        def __init__(self, pipeline, min_price=100):
            self.pipeline = pipeline
            self.min_price = min_price
        
        def predict(self, X):
            """Get predictions and ensure they're positive."""
            predictions = self.pipeline.predict(X)
            return np.maximum(predictions, self.min_price)
    
    # Create the model with the minimum price threshold
    final_model = PositivePriceModel(pipeline)
    
    # Save the model
    print("\nSaving model...")
    os.makedirs('./models', exist_ok=True)
    joblib.dump(final_model, './models/fixed_price_model.joblib')
    
    print("\nFixed price model saved to ./models/fixed_price_model.joblib")
    print("Now update app.py to use this model for predictions.")

if __name__ == "__main__":
    main()
