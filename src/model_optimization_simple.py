#!/usr/bin/env python3
"""
ML Marketplace Model Optimizer (Simple Version)

This script provides a streamlined approach to optimize the price prediction model before UI development:
1. Enhanced feature engineering for hardware components (SSD, screen size, etc.)
2. Hyperparameter tuning with RandomizedSearchCV
3. Model evaluation and selection
4. Feature importance analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
import joblib
import time
from typing import Dict, Any
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "optimized_model")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting model optimization process...")

def load_and_prepare_data(train_path=None, test_path=None, target_col='Precio_Rango'):
    """
    Load and prepare data for model optimization.
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
    
    # Get numeric features only
    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    X_train_numeric = X_train[numeric_cols]
    X_test_numeric = X_test[numeric_cols]
    
    # Handle missing values in numeric data
    for col in numeric_cols:
        # Fill missing values with median
        X_train_numeric[col] = X_train_numeric[col].fillna(X_train_numeric[col].median())
        X_test_numeric[col] = X_test_numeric[col].fillna(X_train_numeric[col].median())
    
    print(f"Using {len(numeric_cols)} numeric features")
    
    return {
        'X_train': X_train_numeric,
        'y_train': y_train,
        'X_test': X_test_numeric,
        'y_test': y_test,
        'feature_names': numeric_cols
    }

def enhance_hardware_features(X_train, X_test):
    """
    Create hardware-specific features for better price prediction.
    """
    print("Enhancing hardware features...")
    
    # Create copies
    X_train_enhanced = X_train.copy()
    X_test_enhanced = X_test.copy()
    
    # Identify hardware features
    hardware_categories = {
        'storage': ['ssd', 'disco', 'storage', 'capacity'],
        'screen': ['screen', 'pantalla', 'display'],
        'ram': ['ram', 'memoria'],
        'processor': ['processor', 'procesador', 'cpu', 'cache', 'caché'],
        'graphics': ['gpu', 'graphic', 'gráfica']
    }
    
    hardware_features = {}
    
    # Find hardware features
    for category, keywords in hardware_categories.items():
        hardware_features[category] = [col for col in X_train.columns 
                                      if any(keyword in col.lower() for keyword in keywords)]
        if hardware_features[category]:
            print(f"Found {len(hardware_features[category])} {category} features")
    
    # Create feature combinations for hardware components
    for category, features in hardware_features.items():
        if len(features) >= 2:
            # Create interaction features
            for i, col1 in enumerate(features[:-1]):
                for col2 in features[i+1:]:
                    interaction_name = f"{category}_{col1}_x_{col2}"
                    X_train_enhanced[interaction_name] = X_train[col1] * X_train[col2]
                    X_test_enhanced[interaction_name] = X_test[col1] * X_test[col2]
    
    # Create specific features for SSD and screen
    ssd_features = hardware_features.get('storage', [])
    screen_features = hardware_features.get('screen', [])
    ram_features = hardware_features.get('ram', [])
    
    # If we have both SSD and screen features, create cross-features
    if ssd_features and screen_features:
        for ssd_col in ssd_features:
            for screen_col in screen_features:
                # Create cross-feature
                feature_name = f"ssd_screen_{ssd_col}_x_{screen_col}"
                X_train_enhanced[feature_name] = X_train[ssd_col] * X_train[screen_col]
                X_test_enhanced[feature_name] = X_test[ssd_col] * X_test[screen_col]
    
    # If we have both SSD and RAM features, create cross-features
    if ssd_features and ram_features:
        for ssd_col in ssd_features:
            for ram_col in ram_features:
                # Create cross-feature
                feature_name = f"ssd_ram_{ssd_col}_x_{ram_col}"
                X_train_enhanced[feature_name] = X_train[ssd_col] * X_train[ram_col]
                X_test_enhanced[feature_name] = X_test[ssd_col] * X_test[ram_col]
    
    # Count new features
    new_features = len(X_train_enhanced.columns) - len(X_train.columns)
    print(f"Created {new_features} new hardware-focused features")
    
    return X_train_enhanced, X_test_enhanced

def optimize_random_forest(X_train, y_train, cv=5, n_iter=20):
    """
    Optimize Random Forest hyperparameters.
    """
    print("Optimizing Random Forest model...")
    
    # Parameter distribution for RandomizedSearchCV
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Create Random Forest model
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # RandomizedSearchCV
    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    # Fit model
    start_time = time.time()
    rf_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Print results
    print(f"Random Forest optimization completed in {training_time:.2f} seconds")
    print(f"Best parameters: {rf_search.best_params_}")
    print(f"Best RMSE: {-rf_search.best_score_:.4f}")
    
    # Get best model
    best_rf = rf_search.best_estimator_
    
    return best_rf, rf_search.cv_results_

def optimize_gradient_boosting(X_train, y_train, cv=5, n_iter=20):
    """
    Optimize Gradient Boosting hyperparameters.
    """
    print("Optimizing Gradient Boosting model...")
    
    # Parameter distribution for RandomizedSearchCV
    param_dist = {
        'n_estimators': randint(50, 300),
        'learning_rate': uniform(0.01, 0.2),
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'subsample': uniform(0.7, 0.3)
    }
    
    # Create Gradient Boosting model
    gb = GradientBoostingRegressor(random_state=42)
    
    # RandomizedSearchCV
    gb_search = RandomizedSearchCV(
        estimator=gb,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    # Fit model
    start_time = time.time()
    gb_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Print results
    print(f"Gradient Boosting optimization completed in {training_time:.2f} seconds")
    print(f"Best parameters: {gb_search.best_params_}")
    print(f"Best RMSE: {-gb_search.best_score_:.4f}")
    
    # Get best model
    best_gb = gb_search.best_estimator_
    
    return best_gb, gb_search.cv_results_

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
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
    """
    print("Analyzing feature importance...")
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save feature importance
    feature_importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_optimized.csv"), index=False)
    
    # Print top 10 features
    print("\nTop 10 Important Features:")
    for i, (feature, importance) in enumerate(zip(feature_importance['feature'][:10], 
                                               feature_importance['importance'][:10]), 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    # Create feature importance plot
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_optimized.png"))
    plt.close()
    
    # Analyze hardware features
    hardware_keywords = ['ssd', 'screen', 'pantalla', 'display', 'ram', 'memoria', 
                       'processor', 'procesador', 'cpu', 'gpu', 'graphic', 'gráfica']
    
    hardware_features = [feature for feature in feature_importance['feature'] 
                        if any(keyword in feature.lower() for keyword in hardware_keywords)]
    
    hardware_importance = feature_importance[feature_importance['feature'].isin(hardware_features)]
    
    if not hardware_importance.empty:
        print("\nHardware Features Importance:")
        for i, (feature, importance) in enumerate(zip(hardware_importance['feature'][:10], 
                                                   hardware_importance['importance'][:10]), 1):
            print(f"  {i}. {feature}: {importance:.4f}")
    
    return feature_importance

def save_optimized_model(model, feature_names):
    """
    Save the optimized model for use in the UI.
    """
    # Create model package
    model_package = {
        'model': model,
        'feature_names': feature_names,
        'creation_date': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save model
    model_path = os.path.join(MODEL_DIR, "optimized_model.joblib")
    joblib.dump(model_package, model_path)
    print(f"Optimized model saved to {model_path}")
    
    return model_path

if __name__ == "__main__":
    # Load and prepare data
    data = load_and_prepare_data()
    
    # Enhance hardware features
    X_train_enhanced, X_test_enhanced = enhance_hardware_features(data['X_train'], data['X_test'])
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enhanced)
    X_test_scaled = scaler.transform(X_test_enhanced)
    
    # Optimize models
    best_rf, rf_results = optimize_random_forest(X_train_scaled, data['y_train'], n_iter=20)
    best_gb, gb_results = optimize_gradient_boosting(X_train_scaled, data['y_train'], n_iter=20)
    
    # Evaluate models
    rf_eval = evaluate_model(best_rf, X_test_scaled, data['y_test'])
    gb_eval = evaluate_model(best_gb, X_test_scaled, data['y_test'])
    
    # Select the best model
    if rf_eval['r2'] > gb_eval['r2']:
        best_model = best_rf
        print("Random Forest selected as the best model")
    else:
        best_model = best_gb
        print("Gradient Boosting selected as the best model")
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(best_model, X_train_enhanced.columns)
    
    # Save optimized model
    model_path = save_optimized_model(best_model, X_train_enhanced.columns.tolist())
    
    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, "feature_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Feature scaler saved to {scaler_path}")
    
    print("\n" + "="*50)
    print("MODEL OPTIMIZATION COMPLETED")
    print("="*50)
    print("\nReady to start developing the UI!")
