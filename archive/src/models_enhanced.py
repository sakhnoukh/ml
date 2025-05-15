#!/usr/bin/env python3
"""
Enhanced ML Marketplace Model with Focus on Key Hardware Features

This module builds on the base model but focuses on emphasizing key hardware features
like SSD storage and screen size, which are critical determinants of computer pricing.

It provides functionality for:
1. Loading preprocessed data
2. Feature enhancement for key hardware specifications
3. Training models with emphasis on important hardware features
4. Evaluating model performance
5. Making predictions with feature importance visualization
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import gc
from typing import Dict, Tuple, List, Union, Optional, Any
import warnings
import math

# Try to import SHAP, but handle if it's not installed
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP package not available. Model explanations will be limited.")

# Root directory and other paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "model_enhanced_results"), exist_ok=True)

def load_data(train_path=None, test_path=None, 
              target_col='Precio_Rango', test_size=0.2, random_state=42) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """
    Load training and testing data from files or create splits if files don't exist.
    
    Args:
        train_path: Path to training data CSV. If None, uses default path.
        test_path: Path to test data CSV. If None, uses default path.
        target_col: Name of the target column.
        test_size: Proportion of data for test set if splitting.
        random_state: Random seed for reproducibility.
        
    Returns:
        Dictionary containing X_train, X_test, y_train, y_test DataFrames/Series.
    """
    # Default paths
    if train_path is None:
        train_path = os.path.join(DATA_DIR, "train_set.csv")
    if test_path is None:
        test_path = os.path.join(DATA_DIR, "test_set.csv")
    
    # Check if train/test files exist
    train_exists = os.path.exists(train_path)
    test_exists = os.path.exists(test_path)
    
    if train_exists and test_exists:
        # Load pre-split data
        print(f"Loading pre-split train/test data from {train_path} and {test_path}")
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        # Separate features and target
        if target_col in train_data.columns and target_col in test_data.columns:
            X_train = train_data.drop(columns=[target_col])
            y_train = train_data[target_col]
            X_test = test_data.drop(columns=[target_col])
            y_test = test_data[target_col]
        else:
            raise ValueError(f"Target column '{target_col}' not found in data.")
    else:
        # Load and split data
        print("Pre-split data not found. Loading from engineered features...")
        data_path = os.path.join(DATA_DIR, "db_computers_features.csv")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        # Load data
        data = pd.read_csv(data_path)
        
        # Separate features and target
        if target_col in data.columns:
            X = data.drop(columns=[target_col])
            y = data[target_col]
        else:
            raise ValueError(f"Target column '{target_col}' not found in data.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Save splits for later
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        train_data.to_csv(os.path.join(DATA_DIR, "train_set.csv"), index=False)
        test_data.to_csv(os.path.join(DATA_DIR, "test_set.csv"), index=False)
    
    # Basic data info
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

def enhance_hardware_features(X_train, X_test, y_train=None):
    """
    Enhance hardware features like SSD capacity and screen size.
    
    Args:
        X_train: Training features DataFrame
        X_test: Test features DataFrame
        y_train: Training target Series (optional, used for correlation)
        
    Returns:
        Enhanced X_train and X_test DataFrames
    """
    print("Enhancing hardware features...")
    
    # Create copies to avoid modifying originals
    X_train_enhanced = X_train.copy()
    X_test_enhanced = X_test.copy()
    
    # List of hardware features to focus on
    key_hardware_features = []
    
    # Look for SSD storage columns
    ssd_columns = [col for col in X_train.columns if 'SSD' in col or 'ssd' in col.lower()]
    if ssd_columns:
        key_hardware_features.extend(ssd_columns)
        print(f"Found SSD storage columns: {ssd_columns}")
    else:
        # Look for general storage columns if specific SSD columns not found
        storage_columns = [col for col in X_train.columns if 'storage' in col.lower() or 
                        'capacity' in col.lower() or 'disco' in col.lower()]
        if storage_columns:
            key_hardware_features.extend(storage_columns)
            print(f"Found storage columns: {storage_columns}")
            
    # Look for screen size columns
    screen_columns = [col for col in X_train.columns if 'screen' in col.lower() or 
                   'pantalla' in col.lower() or 'display' in col.lower()]
    if screen_columns:
        key_hardware_features.extend(screen_columns)
        print(f"Found screen size columns: {screen_columns}")
        
    # Look for RAM columns
    ram_columns = [col for col in X_train.columns if 'RAM' in col or 'ram' in col.lower() or 'memoria' in col.lower()]
    if ram_columns:
        key_hardware_features.extend(ram_columns)
        print(f"Found RAM columns: {ram_columns}")
    
    # Remove duplicates
    key_hardware_features = list(set(key_hardware_features))
    
    if key_hardware_features:
        print(f"Using {len(key_hardware_features)} key hardware features")
        
        # Impute missing values in key hardware features
        imputer = SimpleImputer(strategy='median')
        
        # Only transform numeric features
        numeric_key_features = []
        for feature in key_hardware_features:
            if feature in X_train.columns and pd.api.types.is_numeric_dtype(X_train[feature]):
                numeric_key_features.append(feature)
        
        if numeric_key_features:
            # Impute missing values
            X_train_imputed = X_train_enhanced.copy()
            X_test_imputed = X_test_enhanced.copy()
            
            X_train_imputed[numeric_key_features] = imputer.fit_transform(X_train[numeric_key_features])
            X_test_imputed[numeric_key_features] = imputer.transform(X_test[numeric_key_features])
            
            # Create polynomial features to emphasize hardware relationships
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            
            try:
                # Apply polynomial features
                poly_features_train = poly.fit_transform(X_train_imputed[numeric_key_features])
                poly_features_test = poly.transform(X_test_imputed[numeric_key_features])
                
                # Create feature names
                feature_names = poly.get_feature_names_out(numeric_key_features)
                
                # Add polynomial features to dataframes
                for i, name in enumerate(feature_names):
                    if i >= len(numeric_key_features):  # Skip original features
                        X_train_enhanced[f'hardware_{name}'] = poly_features_train[:, i]
                        X_test_enhanced[f'hardware_{name}'] = poly_features_test[:, i]
                
                print(f"Added {len(feature_names) - len(numeric_key_features)} hardware interaction features")
            except Exception as e:
                print(f"Error creating polynomial features: {e}")
    else:
        print("No specific hardware features found")
        
    # Handle categorical columns related to hardware
    categorical_hardware = [col for col in X_train.columns if ('SSD' in col or 
                                                          'screen' in col.lower() or
                                                          'pantalla' in col.lower() or
                                                          'storage' in col.lower()) and 
                                                         pd.api.types.is_object_dtype(X_train[col])]
    
    for col in categorical_hardware:
        # Convert to numeric if possible or create dummy variables
        try:
            X_train_enhanced[f"{col}_numeric"] = pd.to_numeric(X_train[col], errors='coerce')
            X_test_enhanced[f"{col}_numeric"] = pd.to_numeric(X_test[col], errors='coerce')
        except:
            print(f"Could not convert {col} to numeric")
    
    return X_train_enhanced, X_test_enhanced

def fill_missing_values(X_train, X_test):
    """
    Fill missing values in DataFrames.
    
    Args:
        X_train: Training features DataFrame
        X_test: Test features DataFrame
        
    Returns:
        DataFrames with missing values filled
    """
    # Create copies
    X_train_filled = X_train.copy()
    X_test_filled = X_test.copy()
    
    # Get numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=['number']).columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_cols),
            ('cat', SimpleImputer(strategy='constant', fill_value='missing'), categorical_cols)
        ]
    )
    
    # Apply transformations
    try:
        if len(numeric_cols) > 0:
            X_train_filled[numeric_cols] = preprocessor.fit_transform(X_train[numeric_cols])
            X_test_filled[numeric_cols] = preprocessor.transform(X_test[numeric_cols])
            
        if len(categorical_cols) > 0:
            X_train_filled[categorical_cols] = preprocessor.fit_transform(X_train[categorical_cols])
            X_test_filled[categorical_cols] = preprocessor.transform(X_test[categorical_cols])
    except Exception as e:
        print(f"Error filling missing values: {e}")
        # Fallback method
        for col in X_train.columns:
            if pd.api.types.is_numeric_dtype(X_train[col]):
                X_train_filled[col] = X_train[col].fillna(X_train[col].median())
                X_test_filled[col] = X_test[col].fillna(X_train[col].median())
            else:
                X_train_filled[col] = X_train[col].fillna('missing')
                X_test_filled[col] = X_test[col].fillna('missing')
    
    return X_train_filled, X_test_filled

def train_enhanced_model(X_train, y_train, cv_folds=5, model_dir=None, random_state=42):
    """
    Train an enhanced RandomForest model with emphasis on hardware features.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training target Series
        cv_folds: Number of cross-validation folds
        model_dir: Directory to save models
        random_state: Random seed for reproducibility
        
    Returns:
        Trained model and feature importance DataFrame
    """
    # Set default model directory
    if model_dir is None:
        model_dir = MODEL_DIR
        
    print("Training enhanced RandomForest model...")
    
    # Fill any remaining missing values
    X_train_filled, _ = fill_missing_values(X_train, X_train)  # Just need training data for now
    
    # Define model with optimized parameters for hardware emphasis
    rf_enhanced = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',  # This helps focus on the most important features
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Train model
    rf_enhanced.fit(X_train_filled, y_train)
    
    # Calculate cross-validation score
    cv_rmse = -np.mean(cross_val_score(
        rf_enhanced, X_train_filled, y_train, 
        cv=cv_folds, 
        scoring='neg_root_mean_squared_error'
    ))
    
    cv_r2 = np.mean(cross_val_score(
        rf_enhanced, X_train_filled, y_train, 
        cv=cv_folds, 
        scoring='r2'
    ))
    
    print(f"Enhanced model CV RMSE: {cv_rmse:.4f}")
    print(f"Enhanced model CV R²: {cv_r2:.4f}")
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_enhanced.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save model and feature importance
    model_path = os.path.join(model_dir, "enhanced_price_model.joblib")
    joblib.dump(rf_enhanced, model_path)
    print(f"Enhanced model saved to {model_path}")
    
    # Save feature importance
    feature_importance.to_csv(os.path.join(OUTPUT_DIR, "model_enhanced_results", "enhanced_feature_importance.csv"), index=False)
    
    # Show hardware features importance
    hardware_features = [col for col in X_train.columns if 
                        'hardware_' in col or 
                        'SSD' in col or 
                        'ssd' in col.lower() or
                        'screen' in col.lower() or
                        'pantalla' in col.lower() or
                        'RAM' in col or
                        'ram' in col.lower()]
    
    hardware_importance = feature_importance[feature_importance['feature'].isin(hardware_features)]
    
    if not hardware_importance.empty:
        print("\nHardware Features Importance:")
        for i, (feature, importance) in enumerate(zip(hardware_importance['feature'], 
                                                   hardware_importance['importance']), 1):
            if i <= 10:  # Show top 10
                print(f"  {i}. {feature}: {importance:.4f}")
    
    return rf_enhanced, feature_importance

def evaluate_enhanced_model(model, X_test, y_test):
    """
    Evaluate enhanced model performance.
    
    Args:
        model: Trained model
        X_test: Test features DataFrame
        y_test: Test target Series
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Fill missing values
    X_test_filled, _ = fill_missing_values(X_test, X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_filled)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Print results
    print("\nEnhanced Model Evaluation:")
    print(f"  Test RMSE: {rmse:.4f}")
    print(f"  Test R²: {r2:.4f}")
    print(f"  Test MAE: {mae:.4f}")
    
    # Return metrics
    return {
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'predictions': y_pred
    }

def predict_with_hardware_focus(model, X_new, hardware_features=None):
    """
    Make predictions with visualization of hardware feature contributions.
    
    Args:
        model: Trained model
        X_new: New data to predict
        hardware_features: List of hardware feature names to highlight
        
    Returns:
        Predictions and feature contributions
    """
    # Fill missing values
    X_new_filled, _ = fill_missing_values(X_new, X_new)
    
    # Make predictions
    predictions = model.predict(X_new_filled)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_new.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # If hardware features not specified, try to detect them
    if hardware_features is None:
        hardware_features = [col for col in X_new.columns if 
                           'hardware_' in col or 
                           'SSD' in col or 
                           'ssd' in col.lower() or
                           'screen' in col.lower() or
                           'pantalla' in col.lower() or
                           'RAM' in col or
                           'ram' in col.lower()]
    
    # Filter importance for hardware features
    hardware_importance = feature_importance[feature_importance['feature'].isin(hardware_features)]
    
    # Return predictions and importance
    return {
        'predictions': predictions,
        'feature_importance': feature_importance,
        'hardware_importance': hardware_importance
    }

if __name__ == "__main__":
    # Load data
    data = load_data(target_col='Precio_Rango')
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    # Enhance hardware features
    X_train_enhanced, X_test_enhanced = enhance_hardware_features(X_train, X_test, y_train)
    
    # Train enhanced model
    enhanced_model, feature_importance = train_enhanced_model(X_train_enhanced, y_train)
    
    # Evaluate model
    evaluation = evaluate_enhanced_model(enhanced_model, X_test_enhanced, y_test)
    
    # Print summary
    print("\n" + "="*50)
    print("ENHANCED MODEL WITH HARDWARE FOCUS SUMMARY")
    print("="*50)
    
    # Show top features
    print("\nTop 10 Important Features:")
    top_features = feature_importance.head(10)
    for i, (feature, importance) in enumerate(zip(top_features['feature'], top_features['importance']), 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    # Show hardware features specifically
    hardware_features = [col for col in X_train_enhanced.columns if 
                       'hardware_' in col or 
                       'SSD' in col or 
                       'ssd' in col.lower() or
                       'screen' in col.lower() or
                       'pantalla' in col.lower() or
                       'RAM' in col or
                       'ram' in col.lower()]
    
    hardware_importance = feature_importance[feature_importance['feature'].isin(hardware_features)]
    
    if not hardware_importance.empty:
        print("\nHardware Features Importance:")
        for i, (feature, importance) in enumerate(zip(hardware_importance['feature'], 
                                               hardware_importance['importance']), 1):
            if i <= 10:  # Show top 10
                print(f"  {i}. {feature}: {importance:.4f}")
    
    print("\nModel Performance:")
    print(f"  Test RMSE: {evaluation['rmse']:.4f}")
    print(f"  Test R²: {evaluation['r2']:.4f}")
    print(f"  Test MAE: {evaluation['mae']:.4f}")
    
    print("\nModel saved to:")
    print(f"  {os.path.join(MODEL_DIR, 'enhanced_price_model.joblib')}")
    print("="*50)
