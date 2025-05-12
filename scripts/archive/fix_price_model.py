"""
Fix Price Model Script

This script fixes the price prediction issues by:
1. Training a simple, stable model (Ridge Regression)
2. Using the PositivePriceScaler to ensure positive predictions
3. Saving the model and scaler for use in the app
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

# Add the src directory to the Python path
sys.path.append('./src')

# Import project modules
from features import engineer_features, identify_core_features, get_feature_names
from positive_price_scaler import PositivePriceScaler

def main():
    """Fix and save a working price prediction model."""
    print("=== Fixing Price Prediction Model ===")
    
    # Load the data
    print("\nLoading data...")
    data_path = "./data/db_computers_2025_processed.csv"
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    
    if 'Price' in df.columns:
        print(f"\nPrice statistics:")
        print(f"Mean: {df['Price'].mean():.2f}")
        print(f"Std Dev: {df['Price'].std():.2f}")
        print(f"Min: {df['Price'].min():.2f}")
        print(f"Max: {df['Price'].max():.2f}")
    
    # Create a positive price scaler
    print("\nCreating positive price scaler...")
    price_scaler = PositivePriceScaler(price_column='Price', min_price=500)
    price_scaler.fit(df)
    
    print(f"Price scaler mean: {price_scaler.price_mean_:.2f}")
    print(f"Price scaler std: {price_scaler.price_std_:.2f}")
    
    # Scale the target variable
    scaled_prices = price_scaler.transform_target(df['Price'])
    
    print(f"\nScaled price statistics:")
    print(f"Mean: {scaled_prices.mean():.4f}")
    print(f"Std Dev: {scaled_prices.std():.4f}")
    print(f"Min: {scaled_prices.min():.4f}")
    print(f"Max: {scaled_prices.max():.4f}")
    
    # Identify features
    numeric_features, categorical_features = identify_core_features(df)
    print(f"\nIdentified {len(numeric_features)} numeric features: {numeric_features}")
    print(f"Identified {len(categorical_features)} categorical features: {categorical_features}")
    
    # Engineer features
    X_transformed, feature_pipeline = engineer_features(
        df,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )
    
    feature_names = get_feature_names(feature_pipeline, numeric_features, categorical_features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, scaled_prices, test_size=0.2, random_state=42
    )
    
    # Train a Ridge model
    print("\nTraining Ridge Regression model...")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"Test RMSE (scaled): {test_rmse:.4f}")
    print(f"Test R² (scaled): {test_r2:.4f}")
    
    # Evaluate in original price scale
    y_test_orig = price_scaler.inverse_transform(None, y_test)
    y_pred_orig = price_scaler.inverse_transform(None, y_pred)
    
    orig_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    orig_mae = mean_absolute_error(y_test_orig, y_pred_orig)
    orig_r2 = r2_score(y_test_orig, y_pred_orig)
    
    print(f"\nOriginal Scale Metrics:")
    print(f"RMSE: {orig_rmse:.2f} €")
    print(f"MAE: {orig_mae:.2f} €")
    print(f"R²: {orig_r2:.4f}")
    
    # Check prediction statistics
    print(f"\nPrediction Statistics (Original Scale):")
    print(f"Min: {min(y_pred_orig):.2f} €")
    print(f"Max: {max(y_pred_orig):.2f} €")
    print(f"Mean: {np.mean(y_pred_orig):.2f} €")
    
    # Test with median values
    print("\nTesting with median values...")
    
    # Create a sample input with median values
    sample_input = {}
    
    # Add numeric features with median values
    for feature in numeric_features:
        if feature.lower() == 'price':
            continue
        sample_input[feature] = df[feature].median()
    
    # Add categorical features with most common values
    for feature in categorical_features:
        sample_input[feature] = df[feature].mode().iloc[0]
    
    # Convert to DataFrame
    input_df = pd.DataFrame([sample_input])
    
    # Add a dummy Price column
    input_df['Price'] = 0
    
    # Transform input
    X_sample = feature_pipeline.transform(input_df)
    
    # Predict price
    predicted_price_scaled = model.predict(X_sample)[0]
    predicted_price = price_scaler.inverse_transform(None, predicted_price_scaled)
    
    print(f"Predicted Price (scaled): {predicted_price_scaled:.4f}")
    print(f"Predicted Price: €{predicted_price:.2f}")
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(models_dir, "price_model.joblib"))
    joblib.dump(price_scaler, os.path.join(models_dir, "price_scaler.joblib"))
    joblib.dump(feature_pipeline, os.path.join(models_dir, "feature_pipeline.joblib"))
    
    print("Model, scaler, and pipeline saved successfully!")
    print("\nYou can now run the Streamlit app with: streamlit run src/app.py")

if __name__ == "__main__":
    main()
