"""
Model Diagnostics Script

This script performs a thorough evaluation of different regression models for
computer price prediction to identify and fix issues with the predictions.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression

# Add the src directory to the Python path
sys.path.append('./src')

# Import project modules
from clean_price import clean_price_column
from features import engineer_features, identify_core_features, get_feature_names
from price_scaler import PriceScaler

def load_data(data_path):
    """Load and inspect the data."""
    print(f"\n=== Loading data from {data_path} ===")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    
    if 'Price' in df.columns:
        print(f"\nPrice statistics:")
        print(f"Mean: {df['Price'].mean():.2f}")
        print(f"Std Dev: {df['Price'].std():.2f}")
        print(f"Min: {df['Price'].min():.2f}")
        print(f"Max: {df['Price'].max():.2f}")
    else:
        print("\nError: 'Price' column not found in the dataset.")
    
    return df

def prepare_features(df):
    """Prepare features and target variables."""
    print("\n=== Preparing features and target variables ===")
    
    # Identify core features
    numeric_features, categorical_features = identify_core_features(df)
    print(f"Identified {len(numeric_features)} numeric features: {numeric_features}")
    print(f"Identified {len(categorical_features)} categorical features: {categorical_features}")
    
    # Engineer features
    X_transformed, feature_pipeline = engineer_features(
        df,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )
    
    # Get feature names
    feature_names = get_feature_names(feature_pipeline, numeric_features, categorical_features)
    print(f"Total features after engineering: {len(feature_names)}")
    
    # Apply price scaling
    price_scaler = PriceScaler(price_column='Price')
    price_scaler.fit(df)
    scaled_prices = price_scaler.transform_target(df['Price'])
    
    print(f"\nPrice statistics before scaling:")
    print(f"Mean: {df['Price'].mean():.2f}")
    print(f"Std Dev: {df['Price'].std():.2f}")
    
    print(f"\nScaled price statistics:")
    print(f"Mean: {scaled_prices.mean():.4f}")
    print(f"Std Dev: {scaled_prices.std():.4f}")
    print(f"Min: {scaled_prices.min():.4f}")
    print(f"Max: {scaled_prices.max():.4f}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, scaled_prices, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_pipeline': feature_pipeline,
        'price_scaler': price_scaler,
        'feature_names': feature_names,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }

def evaluate_model(model, data, name):
    """Evaluate a model and calculate performance metrics."""
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    price_scaler = data['price_scaler']
    
    # Train the model
    print(f"\n=== Training {name} model ===")
    model.fit(X_train, y_train)
    
    # Make predictions on training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics on training set
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Calculate metrics on test set
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmse = np.sqrt(-cross_val_score(
        model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv
    ))
    
    # Evaluate predictions in original price scale
    y_test_orig = price_scaler.inverse_transform(None, y_test)
    y_test_pred_orig = price_scaler.inverse_transform(None, y_test_pred)
    orig_rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred_orig))
    orig_mae = mean_absolute_error(y_test_orig, y_test_pred_orig)
    orig_r2 = r2_score(y_test_orig, y_test_pred_orig)
    
    # Print results
    print(f"\n{name} Model Performance:")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Cross-validation RMSE: {cv_rmse.mean():.4f} (±{cv_rmse.std():.4f})")
    
    print(f"\nOriginal Price Scale Metrics:")
    print(f"RMSE: {orig_rmse:.2f} €")
    print(f"MAE: {orig_mae:.2f} €")
    print(f"R²: {orig_r2:.4f}")
    
    # Check for negative predictions
    neg_count = sum(y_test_pred_orig < 0)
    neg_percent = 100 * neg_count / len(y_test_pred_orig)
    print(f"\nNegative Predictions: {neg_count} ({neg_percent:.2f}%)")
    
    # Print prediction statistics
    print(f"\nPrediction Statistics (Original Scale):")
    print(f"Min: {y_test_pred_orig.min():.2f} €")
    print(f"Max: {y_test_pred_orig.max():.2f} €")
    print(f"Mean: {y_test_pred_orig.mean():.2f} €")
    print(f"Std Dev: {y_test_pred_orig.std():.2f} €")
    
    # Return the trained model and metrics
    return {
        'model': model,
        'metrics': {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_rmse': cv_rmse.mean(),
            'orig_rmse': orig_rmse,
            'orig_mae': orig_mae,
            'orig_r2': orig_r2,
            'neg_count': neg_count,
            'neg_percent': neg_percent
        }
    }

def test_sample_predictions(results, data):
    """Test models with sample inputs."""
    X_test = data['X_test']
    y_test = data['y_test']
    price_scaler = data['price_scaler']
    feature_pipeline = data['feature_pipeline']
    
    print("\n=== Testing with Sample Inputs ===")
    
    # Create a sample input with median values
    print("\nTesting with median values input:")
    
    # Load the original data
    df = load_data('./data/db_computers_2025_processed.csv')
    
    # Create a sample input dictionary with median values
    sample_input = {}
    
    # Get numeric and categorical features
    numeric_features = data['numeric_features']
    categorical_features = data['categorical_features']
    
    # Add numeric features with median values
    for feature in numeric_features:
        if feature.lower() == 'price':
            # Skip the price feature as it's what we're predicting
            continue
        sample_input[feature] = df[feature].median()
    
    # Add categorical features with most common values
    for feature in categorical_features:
        sample_input[feature] = df[feature].mode().iloc[0]
    
    # Convert to DataFrame
    input_df = pd.DataFrame([sample_input])
    
    # Add a dummy Price column for the feature pipeline
    input_df['Price'] = 0
    
    # Transform input using feature pipeline
    X_sample = feature_pipeline.transform(input_df)
    
    # Make predictions with each model
    print("\nPredictions for sample input with median values:")
    for model_name, result in results.items():
        # Get the model
        model = result['model']
        
        # Predict price (scaled)
        predicted_price_scaled = model.predict(X_sample)[0]
        
        # Convert to original price scale
        predicted_price = price_scaler.inverse_transform(None, predicted_price_scaled)
        
        print(f"{model_name}:")
        print(f"  Predicted Price (scaled): {predicted_price_scaled:.4f}")
        print(f"  Predicted Price: €{predicted_price:.2f}")

def fix_price_scaler(data):
    """Create a modified price scaler that always returns positive values."""
    original_scaler = data['price_scaler']
    
    # Create a new PriceScaler with modified inverse_transform
    class PositivePriceScaler(PriceScaler):
        def inverse_transform(self, X, y_pred):
            """Ensure predictions are always positive and reasonable."""
            if self.price_mean_ is not None and self.price_std_ is not None:
                # Basic inverse transform
                raw_price = y_pred * self.price_std_ + self.price_mean_
                
                # Ensure price is positive and reasonable
                return np.maximum(raw_price, 100)
            else:
                return np.maximum(y_pred, 100)
    
    # Create and fit the new scaler
    positive_scaler = PositivePriceScaler(price_column='Price')
    positive_scaler.price_mean_ = original_scaler.price_mean_
    positive_scaler.price_std_ = original_scaler.price_std_
    
    print("\n=== Fixed Price Scaler Created ===")
    print(f"Price Mean: {positive_scaler.price_mean_:.2f}")
    print(f"Price Std Dev: {positive_scaler.price_std_:.2f}")
    
    return positive_scaler

def save_best_model(results, data, models_dir="./models"):
    """Identify and save the best model."""
    print("\n=== Saving Best Model ===")
    
    # Create metrics DataFrame
    metrics = {name: result['metrics'] for name, result in results.items()}
    metrics_df = pd.DataFrame(metrics).T
    
    # Find best model based on test RMSE
    best_model_name = metrics_df['test_rmse'].idxmin()
    best_model = results[best_model_name]['model']
    
    print(f"Best model: {best_model_name}")
    print(f"Test RMSE: {metrics_df.loc[best_model_name, 'test_rmse']:.4f}")
    print(f"Original Scale RMSE: €{metrics_df.loc[best_model_name, 'orig_rmse']:.2f}")
    
    # Fix the price scaler
    fixed_scaler = fix_price_scaler(data)
    
    # Save the best model and fixed price scaler
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(models_dir, "best_price_model.joblib"))
    joblib.dump(fixed_scaler, os.path.join(models_dir, "best_price_scaler.joblib"))
    joblib.dump(data['feature_pipeline'], os.path.join(models_dir, "best_feature_pipeline.joblib"))
    
    print(f"\nBest model, fixed price scaler, and feature pipeline saved to {models_dir}.")

def validate_fixed_model(model, data, fixed_scaler):
    """Validate the model with the fixed price scaler."""
    print("\n=== Validating Fixed Model ===")
    
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Make predictions
    y_pred_scaled = model.predict(X_test)
    
    # Convert to original price scale using fixed scaler
    y_pred_fixed = fixed_scaler.inverse_transform(None, y_pred_scaled)
    
    # Convert actual values to original scale
    y_test_orig = fixed_scaler.inverse_transform(None, y_test)
    
    # Calculate metrics
    fixed_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_fixed))
    fixed_mae = mean_absolute_error(y_test_orig, y_pred_fixed)
    fixed_r2 = r2_score(y_test_orig, y_pred_fixed)
    
    print(f"Fixed Model Performance:")
    print(f"RMSE: {fixed_rmse:.2f} €")
    print(f"MAE: {fixed_mae:.2f} €")
    print(f"R²: {fixed_r2:.4f}")
    
    # Check for negative predictions
    neg_count = sum(y_pred_fixed < 0)
    print(f"Negative Predictions: {neg_count} (should be 0)")
    
    # Print prediction statistics
    print(f"\nFixed Prediction Statistics:")
    print(f"Min: {y_pred_fixed.min():.2f} €")
    print(f"Max: {y_pred_fixed.max():.2f} €")
    print(f"Mean: {y_pred_fixed.mean():.2f} €")
    print(f"Std Dev: {y_pred_fixed.std():.2f} €")

def main():
    """Run the full model diagnostics process."""
    print("=== Computer Price Prediction Model Diagnostics ===")
    
    # Define paths
    data_path = "./data/db_computers_2025_processed.csv"
    models_dir = "./models"
    
    # Load the data
    df = load_data(data_path)
    
    # Prepare features and target variables
    data = prepare_features(df)
    
    # Define models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, data, name)
    
    # Test with sample inputs
    test_sample_predictions(results, data)
    
    # Save the best model with a fixed price scaler
    save_best_model(results, data, models_dir)
    
    # Validate the fixed model
    best_model_name = min(results, key=lambda x: results[x]['metrics']['test_rmse'])
    best_model = results[best_model_name]['model']
    fixed_scaler = fix_price_scaler(data)
    validate_fixed_model(best_model, data, fixed_scaler)
    
    print("\n=== Diagnostics Complete ===")
    print("The best model and fixed price scaler have been saved to the models directory.")
    print("To use the new model, run the Streamlit app with: streamlit run src/app.py")

if __name__ == "__main__":
    main()
