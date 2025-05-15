#!/usr/bin/env python3
"""
Feature Engineering for ML Marketplace

This module provides streamlined feature engineering functionality for the ML Marketplace
project, with a focus on hardware features (especially SSD and screen size) that
significantly impact computer price prediction.

Functions:
    load_data: Load training and testing data
    create_hardware_features: Generate hardware-specific features with focus on SSD and screen
    scale_features: Standardize numeric features
    select_important_features: Choose the most predictive features
    process_data_pipeline: Full data processing pipeline
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
from typing import Dict, List, Tuple, Any

# Set up directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

def load_data(train_path=None, test_path=None, target_col='Precio_Rango'):
    """
    Load training and testing data.
    
    Args:
        train_path: Path to training data file (default: data/train_set.csv)
        test_path: Path to testing data file (default: data/test_set.csv)
        target_col: Target column name
        
    Returns:
        Dictionary with data components
    """
    # Default paths
    if train_path is None:
        train_path = os.path.join(DATA_DIR, "train_set.csv")
    if test_path is None:
        test_path = os.path.join(DATA_DIR, "test_set.csv")
    
    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Separate features and target
    X_train = train_data.drop(columns=[target_col]) if target_col in train_data.columns else train_data
    y_train = train_data[target_col] if target_col in train_data.columns else None
    
    X_test = test_data.drop(columns=[target_col]) if target_col in test_data.columns else test_data
    y_test = test_data[target_col] if target_col in test_data.columns else None
    
    # Get numeric columns
    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    X_train_numeric = X_train[numeric_cols]
    X_test_numeric = X_test[numeric_cols]
    
    # Clean up data
    for col in numeric_cols:
        X_train_numeric[col] = X_train_numeric[col].fillna(X_train_numeric[col].median())
        X_test_numeric[col] = X_test_numeric[col].fillna(X_train_numeric[col].median())
    
    return {
        'X_train': X_train,
        'X_train_numeric': X_train_numeric,
        'y_train': y_train,
        'X_test': X_test,
        'X_test_numeric': X_test_numeric,
        'y_test': y_test,
        'numeric_cols': numeric_cols
    }

def create_hardware_features(X_train, X_test):
    """
    Create hardware-specific features, with emphasis on SSD and screen.
    
    Args:
        X_train: Training features DataFrame
        X_test: Testing features DataFrame
        
    Returns:
        DataFrames with added hardware features
    """
    # Create copies
    X_train_hw = X_train.copy()
    X_test_hw = X_test.copy()
    
    # Identify hardware feature groups
    hardware_categories = {
        'storage': ['ssd', 'disco', 'storage', 'capacity', 'almacenamiento'],
        'screen': ['screen', 'pantalla', 'display', 'resolución', 'resolution'],
        'ram': ['ram', 'memoria'],
        'processor': ['processor', 'procesador', 'cpu', 'cache', 'caché'],
        'graphics': ['gpu', 'graphic', 'gráfica', 'tarjeta']
    }
    
    # Find all hardware features
    hardware_features = {}
    for category, keywords in hardware_categories.items():
        hardware_features[category] = [col for col in X_train.columns 
                                     if any(keyword in col.lower() for keyword in keywords)]
    
    # Create feature interactions for important hardware components
    # Focus on SSD and screen size interactions
    ssd_features = hardware_features.get('storage', [])
    screen_features = hardware_features.get('screen', [])
    ram_features = hardware_features.get('ram', [])
    
    # Create SSD and screen interactions
    if ssd_features and screen_features:
        for ssd_col in ssd_features:
            for screen_col in screen_features:
                # Only for numeric columns
                if (pd.api.types.is_numeric_dtype(X_train[ssd_col]) and 
                    pd.api.types.is_numeric_dtype(X_train[screen_col])):
                    # Create interaction feature
                    feature_name = f"ssd_screen_{ssd_col}_{screen_col}"
                    X_train_hw[feature_name] = X_train[ssd_col] * X_train[screen_col]
                    X_test_hw[feature_name] = X_test[ssd_col] * X_test[screen_col]
    
    # Create SSD and RAM interactions
    if ssd_features and ram_features:
        for ssd_col in ssd_features:
            for ram_col in ram_features:
                # Only for numeric columns
                if (pd.api.types.is_numeric_dtype(X_train[ssd_col]) and 
                    pd.api.types.is_numeric_dtype(X_train[ram_col])):
                    # Create interaction feature
                    feature_name = f"ssd_ram_{ssd_col}_{ram_col}"
                    X_train_hw[feature_name] = X_train[ssd_col] * X_train[ram_col]
                    X_test_hw[feature_name] = X_test[ssd_col] * X_test[ram_col]
    
    # Create within-category interactions for each hardware component
    for category, features in hardware_features.items():
        numeric_features = [col for col in features if pd.api.types.is_numeric_dtype(X_train[col])]
        
        if len(numeric_features) >= 2:
            for i, col1 in enumerate(numeric_features[:-1]):
                for col2 in numeric_features[i+1:]:
                    feature_name = f"{category}_{col1}_{col2}"
                    X_train_hw[feature_name] = X_train[col1] * X_train[col2]
                    X_test_hw[feature_name] = X_test[col1] * X_test[col2]
    
    # Count new features
    new_features = [col for col in X_train_hw.columns if col not in X_train.columns]
    
    return X_train_hw, X_test_hw, new_features

def scale_features(X_train, X_test, save_scaler=True):
    """
    Scale numeric features.
    
    Args:
        X_train: Training features DataFrame
        X_test: Testing features DataFrame
        save_scaler: Whether to save the scaler
        
    Returns:
        Scaled DataFrames and scaler
    """
    # Create scaler
    scaler = StandardScaler()
    
    # Fit and transform
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    # Save scaler
    if save_scaler:
        joblib.dump(scaler, os.path.join(MODEL_DIR, "feature_scaler.joblib"))
    
    return X_train_scaled, X_test_scaled, scaler

def select_important_features(X_train, X_test, y_train, k=None):
    """
    Select the most important features for prediction.
    
    Args:
        X_train: Training features DataFrame
        X_test: Testing features DataFrame
        y_train: Target variable
        k: Number of features to select (defaults to 80% of features)
        
    Returns:
        DataFrames with selected features
    """
    # Default to 80% of features if k not specified
    if k is None:
        k = int(X_train.shape[1] * 0.8)
    
    # Create selector
    selector = SelectKBest(f_regression, k=k)
    
    # Fit and transform
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = X_train.columns[selected_indices]
    
    # Create DataFrames
    X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)
    
    return X_train_selected_df, X_test_selected_df, selected_features

def process_data_pipeline(train_path=None, test_path=None, target_col='Precio_Rango'):
    """
    Full data processing pipeline.
    
    Args:
        train_path: Path to training data
        test_path: Path to testing data
        target_col: Target column name
        
    Returns:
        Dictionary with processed data
    """
    # Step 1: Load data
    data = load_data(train_path, test_path, target_col)
    
    # Step 2: Create hardware features
    X_train_hw, X_test_hw, new_features = create_hardware_features(
        data['X_train_numeric'], 
        data['X_test_numeric']
    )
    
    # Step 3: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_hw, X_test_hw)
    
    # Step 4: Select important features
    X_train_selected, X_test_selected, selected_features = select_important_features(
        X_train_scaled, 
        X_test_scaled, 
        data['y_train']
    )
    
    # Return processed data
    return {
        'X_train': X_train_selected,
        'X_test': X_test_selected,
        'y_train': data['y_train'],
        'y_test': data['y_test'],
        'scaler': scaler,
        'selected_features': selected_features,
        'new_features': new_features
    }

if __name__ == "__main__":
    # Process data
    processed_data = process_data_pipeline()
    
    # Print summary
    print(f"Processed {processed_data['X_train'].shape[0]} training samples")
    print(f"Created {len(processed_data['new_features'])} new hardware features")
    print(f"Selected {len(processed_data['selected_features'])} important features")
    print("Data processing complete and ready for modeling")
