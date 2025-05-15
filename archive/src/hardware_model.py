#!/usr/bin/env python3
"""
Hardware-Focused ML Marketplace Model

This module creates a specialized model that highlights the importance of key hardware
features like SSD capacity and screen size in determining computer prices.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import gc
import warnings
from typing import Dict, Any

# Root directory and paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "hardware_model")

# Create output directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_prepare_data(train_path=None, test_path=None, target_col='Precio_Rango'):
    """
    Load data and prepare it for hardware-focused modeling.
    
    Args:
        train_path: Path to training data CSV file
        test_path: Path to test data CSV file
        target_col: Target column name
        
    Returns:
        Dictionary with prepared data
    """
    # Default paths
    if train_path is None:
        train_path = os.path.join(DATA_DIR, "train_set.csv")
    if test_path is None:
        test_path = os.path.join(DATA_DIR, "test_set.csv")
    
    print(f"Loading data from {train_path} and {test_path}")
    
    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Separate features and target
    X_train = train_data.drop(columns=[target_col]) if target_col in train_data.columns else train_data
    y_train = train_data[target_col] if target_col in train_data.columns else None
    
    X_test = test_data.drop(columns=[target_col]) if target_col in test_data.columns else test_data
    y_test = test_data[target_col] if target_col in test_data.columns else None
    
    print(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Find hardware features
    hardware_features = find_hardware_features(X_train)
    
    # Handle missing values - simply fill them to avoid processing issues
    X_train = handle_missing_values(X_train)
    X_test = handle_missing_values(X_test)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'hardware_features': hardware_features
    }

def find_hardware_features(df):
    """
    Find hardware-related features in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary of hardware feature groups
    """
    # Initialize dictionary for different hardware feature groups
    hardware_features = {
        'storage': [],
        'screen': [],
        'ram': [],
        'processor': [],
        'graphics': []
    }
    
    # Storage features (including SSD)
    storage_patterns = ['ssd', 'storage', 'capacity', 'disco', 'almacenamiento']
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in storage_patterns):
            hardware_features['storage'].append(col)
            print(f"Found storage feature: {col}")
    
    # Screen features
    screen_patterns = ['screen', 'pantalla', 'display', 'resolution', 'resolución']
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in screen_patterns):
            hardware_features['screen'].append(col)
            print(f"Found screen feature: {col}")
    
    # RAM features
    ram_patterns = ['ram', 'memoria']
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in ram_patterns):
            hardware_features['ram'].append(col)
            print(f"Found RAM feature: {col}")
    
    # Processor features
    processor_patterns = ['processor', 'procesador', 'cpu', 'core', 'núcleo']
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in processor_patterns):
            hardware_features['processor'].append(col)
            print(f"Found processor feature: {col}")
    
    # Graphics features
    graphics_patterns = ['graphics', 'gpu', 'gráfica', 'video']
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in graphics_patterns):
            hardware_features['graphics'].append(col)
            print(f"Found graphics feature: {col}")
    
    # Summary
    all_hardware = []
    for category, features in hardware_features.items():
        all_hardware.extend(features)
    
    all_hardware = list(set(all_hardware))  # Remove duplicates
    print(f"Found {len(all_hardware)} hardware-related features in total")
    
    return hardware_features

def handle_missing_values(df):
    """
    Simple function to handle missing values without complex imputation.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing values handled
    """
    df_clean = df.copy()
    
    # For numeric columns, fill missing values with median
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # For categorical columns, fill with 'unknown'
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna('unknown')
    
    return df_clean

def train_hardware_focused_model(X_train, y_train, hardware_features, random_state=42):
    """
    Train a model with focus on hardware features.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training target Series
        hardware_features: Dictionary of hardware feature groups
        random_state: Random seed for reproducibility
        
    Returns:
        Trained model
    """
    print("Training hardware-focused RandomForest model...")
    
    # Combine all hardware features
    all_hardware = []
    for category, features in hardware_features.items():
        all_hardware.extend(features)
    
    all_hardware = list(set(all_hardware))  # Remove duplicates
    
    # Check if hardware features exist in the data
    existing_hardware = [col for col in all_hardware if col in X_train.columns]
    print(f"Using {len(existing_hardware)} hardware features for focused model")
    
    # Create model with optimized parameters
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Train model - if exceptions occur, handle them gracefully
    try:
        rf_model.fit(X_train, y_train)
        
        # Calculate cross-validation scores
        cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
        print(f"Cross-validation R² scores: {cv_scores}")
        print(f"Mean CV R²: {np.mean(cv_scores):.4f}")
        
        # Save model
        model_path = os.path.join(MODEL_DIR, "hardware_model.joblib")
        joblib.dump(rf_model, model_path)
        print(f"Model saved to {model_path}")
        
        return rf_model
    except Exception as e:
        print(f"Error training model: {e}")
        print("Trying with a subset of features...")
        
        # Try with just numeric features
        numeric_cols = X_train.select_dtypes(include=['number']).columns
        print(f"Using {len(numeric_cols)} numeric features")
        
        rf_model.fit(X_train[numeric_cols], y_train)
        
        # Save model
        model_path = os.path.join(MODEL_DIR, "hardware_model_numeric_only.joblib")
        joblib.dump(rf_model, model_path)
        print(f"Model saved to {model_path}")
        
        return rf_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features DataFrame
        y_test: Test target Series
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"Error predicting with full feature set: {e}")
        print("Using numeric features only...")
        
        # Try with just numeric features
        numeric_cols = X_test.select_dtypes(include=['number']).columns
        y_pred = model.predict(X_test[numeric_cols])
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Print metrics
    print("\nModel Evaluation:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    return {
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'predictions': y_pred
    }

def analyze_hardware_importance(model, X_train, hardware_features):
    """
    Analyze the importance of hardware features in the model.
    
    Args:
        model: Trained model
        X_train: Training features DataFrame
        hardware_features: Dictionary of hardware feature groups
        
    Returns:
        DataFrame with feature importance
    """
    print("\nAnalyzing hardware feature importance...")
    
    # Get feature importance
    try:
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    except Exception as e:
        print(f"Error getting feature importance for all features: {e}")
        print("Using numeric features only...")
        
        # Try with just numeric features
        numeric_cols = X_train.select_dtypes(include=['number']).columns
        feature_importance = pd.DataFrame({
            'feature': numeric_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    # Save feature importance
    feature_importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    
    # Analyze importance for each hardware category
    hardware_importance = {}
    
    for category, features in hardware_features.items():
        # Filter features that exist in the dataframe
        existing_features = [f for f in features if f in feature_importance['feature'].values]
        
        if existing_features:
            # Calculate total importance for this category
            category_importance = feature_importance[feature_importance['feature'].isin(existing_features)]
            total_importance = category_importance['importance'].sum()
            
            hardware_importance[category] = {
                'features': existing_features,
                'importance': total_importance,
                'detail': category_importance
            }
            
            print(f"\n{category.upper()} Features Importance (Total: {total_importance:.4f}):")
            for i, (feature, importance) in enumerate(zip(category_importance['feature'], 
                                                     category_importance['importance']), 1):
                if i <= 5:  # Show top 5
                    print(f"  {i}. {feature}: {importance:.4f}")
    
    # Create a summary of hardware importance
    summary = pd.DataFrame({
        'hardware_category': list(hardware_importance.keys()),
        'total_importance': [info['importance'] for info in hardware_importance.values()],
        'feature_count': [len(info['features']) for info in hardware_importance.values()]
    }).sort_values('total_importance', ascending=False)
    
    # Save summary
    summary.to_csv(os.path.join(OUTPUT_DIR, "hardware_importance_summary.csv"), index=False)
    
    # Create bar chart of hardware category importance
    plt.figure(figsize=(10, 6))
    plt.bar(summary['hardware_category'], summary['total_importance'])
    plt.title('Importance of Hardware Categories for Price Prediction')
    plt.xlabel('Hardware Category')
    plt.ylabel('Total Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "hardware_importance.png"))
    plt.close()
    
    return feature_importance, hardware_importance

def predict_with_explanation(model, new_data, hardware_features):
    """
    Make predictions and explain the contribution of hardware components.
    
    Args:
        model: Trained model
        new_data: New data to predict
        hardware_features: Dictionary of hardware feature groups
        
    Returns:
        Dictionary with predictions and explanations
    """
    # Handle missing values
    new_data_clean = handle_missing_values(new_data)
    
    # Make predictions
    try:
        predictions = model.predict(new_data_clean)
    except Exception as e:
        print(f"Error predicting with full feature set: {e}")
        print("Using numeric features only...")
        
        # Try with just numeric features
        numeric_cols = new_data_clean.select_dtypes(include=['number']).columns
        predictions = model.predict(new_data_clean[numeric_cols])
    
    # For each hardware category, calculate its contribution
    feature_importance = pd.DataFrame({
        'feature': model.feature_names_in_,
        'importance': model.feature_importances_
    })
    
    hardware_contribution = {}
    for category, features in hardware_features.items():
        # Filter features that exist in the model
        existing_features = [f for f in features if f in feature_importance['feature'].values]
        
        if existing_features:
            # Calculate total importance for this category
            category_importance = feature_importance[feature_importance['feature'].isin(existing_features)]
            total_importance = category_importance['importance'].sum()
            
            hardware_contribution[category] = {
                'importance': total_importance,
                'contribution_pct': total_importance * 100  # As percentage
            }
    
    return {
        'predictions': predictions,
        'hardware_contribution': hardware_contribution
    }

if __name__ == "__main__":
    # Load and prepare data
    data = load_and_prepare_data()
    
    # Train hardware-focused model
    model = train_hardware_focused_model(
        data['X_train'], 
        data['y_train'],
        data['hardware_features']
    )
    
    # Evaluate model
    evaluation = evaluate_model(model, data['X_test'], data['y_test'])
    
    # Analyze hardware importance
    feature_importance, hardware_importance = analyze_hardware_importance(
        model, 
        data['X_train'],
        data['hardware_features']
    )
    
    # Print summary
    print("\n" + "="*50)
    print("HARDWARE-FOCUSED MODEL SUMMARY")
    print("="*50)
    print("\nModel Performance:")
    print(f"  RMSE: {evaluation['rmse']:.4f}")
    print(f"  R²: {evaluation['r2']:.4f}")
    print(f"  MAE: {evaluation['mae']:.4f}")
    
    # Print the relative importance of hardware categories
    if hardware_importance:
        summary = pd.DataFrame({
            'hardware_category': list(hardware_importance.keys()),
            'total_importance': [info['importance'] for info in hardware_importance.values()]
        }).sort_values('total_importance', ascending=False)
        
        print("\nHardware Category Importance:")
        for i, (category, importance) in enumerate(zip(summary['hardware_category'], 
                                                  summary['total_importance']), 1):
            print(f"  {i}. {category}: {importance:.4f} ({importance*100:.2f}%)")
    
    print("\nModel and analysis files saved to:")
    print(f"  {os.path.join(MODEL_DIR, 'hardware_model.joblib')}")
    print(f"  {OUTPUT_DIR}")
    print("="*50)
