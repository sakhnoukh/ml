#!/usr/bin/env python3
"""
ML Marketplace Model

This module provides streamlined model training, evaluation, and prediction
functionality for the ML Marketplace project. It works with the processed data
from feature_engineering.py to build optimized models for computer price prediction.

Functions:
    train_optimized_model: Train model with optimized hyperparameters
    evaluate_model: Evaluate model performance
    analyze_feature_importance: Analyze and visualize important features
    save_model: Save trained model
    load_model: Load saved model
    predict: Make predictions using the trained model
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import joblib
import time
from typing import Dict, Any, Union, List

# Set up directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "model_results"), exist_ok=True)

def train_optimized_model(X_train, y_train, model_type='gradient_boosting'):
    """
    Train model with optimized hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to train ('gradient_boosting' or 'random_forest')
        
    Returns:
        Trained model
    """
    print(f"Training optimized {model_type} model...")
    
    if model_type == 'gradient_boosting':
        # Optimized Gradient Boosting parameters
        model = GradientBoostingRegressor(
            n_estimators=171,
            learning_rate=0.085,
            max_depth=7,
            min_samples_split=8,
            min_samples_leaf=8,
            subsample=0.75,
            random_state=42
        )
    else:
        # Optimized Random Forest parameters
        model = RandomForestRegressor(
            n_estimators=149,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=1,
            max_features='log2',
            random_state=42,
            n_jobs=-1
        )
    
    # Train model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Model trained in {training_time:.2f} seconds")
    
    # Calculate cross-validation score
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=5, 
        scoring='r2'
    )
    
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {np.mean(cv_scores):.4f}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Print results
    print(f"Model Evaluation:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    return {
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'predictions': y_pred
    }

def analyze_feature_importance(model, feature_names):
    """
    Analyze and visualize feature importance.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance
    """
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save feature importance
    feature_importance.to_csv(os.path.join(OUTPUT_DIR, "model_results", "feature_importance.csv"), index=False)
    
    # Print top 10 features
    print("\nTop 10 Important Features:")
    for i, (feature, importance) in enumerate(zip(feature_importance['feature'][:10], 
                                               feature_importance['importance'][:10]), 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    # Create feature importance plot
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(10)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_results", "feature_importance.png"))
    plt.close()
    
    return feature_importance

def save_model(model, feature_names):
    """
    Save model and metadata.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        Path to saved model
    """
    # Create model package
    model_package = {
        'model': model,
        'feature_names': feature_names,
        'creation_date': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save model
    model_path = os.path.join(MODEL_DIR, "price_model.joblib")
    joblib.dump(model_package, model_path)
    print(f"Model saved to {model_path}")
    
    return model_path

def load_model(model_path=None):
    """
    Load saved model.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded model package
    """
    # Default path
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "price_model.joblib")
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load model
    model_package = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    print(f"Model created on: {model_package['creation_date']}")
    
    return model_package

def predict(model, features, scaler=None):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained model or model package
        features: Features to predict on
        scaler: Scaler to use for preprocessing
        
    Returns:
        Predictions
    """
    # Handle model package
    if isinstance(model, dict) and 'model' in model:
        model = model['model']
    
    # Scale features if scaler provided
    if scaler is not None:
        features = scaler.transform(features)
    
    # Make predictions
    predictions = model.predict(features)
    
    return predictions

def get_hardware_importance(feature_importance, include_keywords=None):
    """
    Extract importance of hardware features.
    
    Args:
        feature_importance: DataFrame with feature importance
        include_keywords: List of keywords to identify hardware features
        
    Returns:
        DataFrame with hardware feature importance
    """
    # Default keywords if not provided
    if include_keywords is None:
        include_keywords = ['ssd', 'screen', 'pantalla', 'display', 'ram', 'memoria', 
                          'processor', 'procesador', 'cpu', 'gpu', 'graphic', 'gráfica']
    
    # Filter hardware features
    hardware_features = [feature for feature in feature_importance['feature'] 
                         if any(keyword in feature.lower() for keyword in include_keywords)]
    
    hardware_importance = feature_importance[feature_importance['feature'].isin(hardware_features)]
    
    return hardware_importance

if __name__ == "__main__":
    # Check if we should load a saved model
    model_path = os.path.join(MODEL_DIR, "optimized_model.joblib")
    
    if os.path.exists(model_path):
        print("Using pre-trained optimized model")
        model_package = load_model(model_path)
        model = model_package['model']
        feature_names = model_package['feature_names']
    else:
        # Import feature_engineering only if needed
        from feature_engineering import process_data_pipeline
        
        # Process data
        print("Processing data...")
        data = process_data_pipeline()
        
        # Train model
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        model = train_optimized_model(X_train, y_train, model_type='gradient_boosting')
        feature_names = X_train.columns
        
        # Evaluate model
        evaluation = evaluate_model(model, X_test, y_test)
        
        # Save model
        save_model(model, feature_names)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(model, feature_names)
    
    # Analyze hardware features specifically
    hardware_importance = get_hardware_importance(feature_importance)
    
    print("\nHardware Features Importance:")
    for i, (feature, importance) in enumerate(zip(hardware_importance['feature'][:10], 
                                               hardware_importance['importance'][:10]), 1):
        print(f"  {i}. {feature}: {importance:.4f}")
