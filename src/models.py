#!/usr/bin/env python3
"""
ML Marketplace - Model Training, Evaluation, and Selection Module

This module provides functionality for:
1. Loading preprocessed data
2. Training multiple regression models through grid search
3. Evaluating model performance
4. Selecting the best model
5. Explaining model predictions using SHAP values
6. Exporting models for production

Usage:
    import models
    
    # Train and select the best model
    best_model, cv_results = models.train_and_select_model(X_train, y_train)
    
    # Evaluate model performance
    evaluation = models.evaluate_model(best_model, X_test, y_test)
    
    # Explain model predictions
    feature_importance, shap_values = models.explain_model(best_model, X_train, X_test)
    
    # Make predictions
    predictions = models.predict_price(best_model, X_new)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
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
os.makedirs(os.path.join(OUTPUT_DIR, "model_results"), exist_ok=True)

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

def train_and_select_model(X_train: pd.DataFrame, y_train: pd.Series, 
                         cv_folds: int = 5, 
                         model_dir: str = None,
                         random_state: int = 42) -> Tuple[Any, pd.DataFrame]:
    """
    Train multiple regression models and select the best one based on cross-validation.
    
    Args:
        X_train: Training features DataFrame.
        y_train: Training target Series.
        cv_folds: Number of cross-validation folds.
        model_dir: Directory to save models and results.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple containing (best_model, cv_results_dataframe).
    """
    # Set default model directory
    if model_dir is None:
        model_dir = MODEL_DIR
    
    # Define candidate models with hyperparameter grids
    models = {
        'rf': (RandomForestRegressor(random_state=random_state), {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }),
        'gb': (GradientBoostingRegressor(random_state=random_state), {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }),
        'mlp': (MLPRegressor(random_state=random_state, max_iter=500), {
            'hidden_layer_sizes': [(50,), (100,), (50, 25)],
            'alpha': [0.0001, 0.001],
            'activation': ['relu', 'tanh']
        })
    }
    
    # Container for results
    cv_results = []
    best_models = {}
    
    # Train each model with grid search CV
    for model_name, (model, param_grid) in models.items():
        print(f"\nTraining {model_name}...")
        
        # Define scoring metric (negative RMSE for minimization)
        scoring = 'neg_root_mean_squared_error'
        
        # Create grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv_folds,
            n_jobs=-1,  # Use all available cores
            verbose=1,
            return_train_score=True
        )
        
        # Fit the grid search
        try:
            grid_search.fit(X_train, y_train)
            
            # Get best estimator and parameters
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            # Store best model
            best_models[model_name] = best_model
            
            # Calculate metrics
            cv_rmse = -grid_search.best_score_  # Negative score to positive RMSE
            cv_r2 = np.mean(cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring='r2'))
            
            # Add to results
            cv_results.append({
                'model': model_name,
                'best_params': best_params,
                'cv_rmse': cv_rmse,
                'cv_r2': cv_r2
            })
            
            print(f"  Best parameters: {best_params}")
            print(f"  CV RMSE: {cv_rmse:.2f}")
            print(f"  CV R²: {cv_r2:.3f}")
            
            # Save model
            model_path = os.path.join(model_dir, f"{model_name}_model.joblib")
            joblib.dump(best_model, model_path)
            print(f"  Model saved to {model_path}")
            
            # Clean up to save memory
            gc.collect()
        
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
    
    # Convert results to DataFrame and sort by CV RMSE
    cv_results_df = pd.DataFrame(cv_results)
    if not cv_results_df.empty:
        cv_results_df = cv_results_df.sort_values('cv_rmse')
        
        # Save results
        cv_results_path = os.path.join(model_dir, "model_comparison.csv")
        cv_results_df.to_csv(cv_results_path, index=False)
        print(f"\nModel comparison saved to {cv_results_path}")
        
        # Get best model
        best_model_name = cv_results_df.iloc[0]['model']
        best_model = best_models[best_model_name]
        print(f"\nBest model: {best_model_name}")
        
        # Save best model separately
        best_model_path = os.path.join(model_dir, "price_model.joblib")
        joblib.dump(best_model, best_model_path)
        print(f"Best model saved to {best_model_path}")
        
        return best_model, cv_results_df
    else:
        raise RuntimeError("No models were successfully trained.")

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained model to evaluate.
        X_test: Test features DataFrame.
        y_test: Test target Series.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate additional metrics
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Print evaluation results
    print("\nModel Evaluation on Test Data:")
    print(f"  Test RMSE: {rmse:.2f}")
    print(f"  Test R²: {r2:.3f}")
    print(f"  Test MAE: {mae:.2f}")
    print(f"  Test MAPE: {mape:.2f}%")
    
    # Return metrics as dictionary
    return {
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'mape': mape
    }

def explain_model(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                output_dir: str = None, n_display: int = 10) -> Dict[str, Any]:
    """
    Explain model predictions using feature importance and SHAP values.
    
    Args:
        model: Trained model to explain.
        X_train: Training features DataFrame (for SHAP explainer).
        X_test: Test features DataFrame (sample for SHAP values).
        output_dir: Directory to save explanation plots.
        n_display: Number of top features to display.
        
    Returns:
        Dictionary containing feature importance and optionally SHAP values.
    """
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "model_results")
    
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    explanation = {}
    
    # Extract feature importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save feature importance
        feature_importance.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
        explanation['feature_importance'] = feature_importance
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(n_display)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importance')
        plt.title(f'Top {n_display} Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importance.png"))
        plt.close()
        
        # Print top features
        print(f"\nTop {n_display} Important Features:")
        for i, (feature, importance) in enumerate(zip(top_features['feature'], top_features['importance']), 1):
            print(f"  {i}. {feature}: {importance:.4f}")
    
    # SHAP analysis if available
    if SHAP_AVAILABLE:
        try:
            # Sample test data for SHAP analysis to reduce computation
            X_test_sample = X_test.sample(min(500, len(X_test)), random_state=42)
            
            # Create SHAP explainer
            print("\nComputing SHAP values...")
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model, X_train)
            else:
                if hasattr(model, 'estimators_'):  # For ensemble models
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.Explainer(model, X_train)
            
            # Compute SHAP values
            shap_values = explainer(X_test_sample)
            explanation['shap_values'] = shap_values
            
            # Create summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test_sample, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "shap_summary.png"))
            plt.close()
            
            # Create bar plot
            plt.figure(figsize=(12, 8))
            shap.plots.bar(shap_values, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "shap_bar.png"))
            plt.close()
            
            print(f"SHAP analysis completed and saved to {output_dir}")
        except Exception as e:
            print(f"Error in SHAP analysis: {e}")
    else:
        print("SHAP package not available. Skipping SHAP analysis.")
    
    return explanation

def predict_price(model: Any, X_new: pd.DataFrame) -> pd.Series:
    """
    Make price predictions for new data.
    
    Args:
        model: Trained model to use for prediction.
        X_new: New features DataFrame.
        
    Returns:
        Series of predicted prices.
    """
    # Predict prices
    predictions = model.predict(X_new)
    
    # Ensure predictions are positive
    predictions = np.maximum(predictions, 0)
    
    return pd.Series(predictions, index=X_new.index, name='predicted_price')

def load_model(model_path: str = None) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to saved model. If None, uses default path.
        
    Returns:
        Loaded model.
    """
    # Set default model path
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "price_model.joblib")
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load model
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    
    return model

if __name__ == "__main__":
    # Load data
    data = load_data(target_col='Precio_Rango')
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    # Train and select the best model
    best_model, cv_results = train_and_select_model(X_train, y_train)
    
    # Evaluate the best model
    evaluation = evaluate_model(best_model, X_test, y_test)
    
    # Explain the model
    explanation = explain_model(best_model, X_train, X_test)
    
    # Summary of results
    print("\n" + "="*50)
    print("ML MARKETPLACE MODEL TRAINING SUMMARY")
    print("="*50)
    print(f"Best model: {cv_results.iloc[0]['model']}")
    print(f"Model parameters: {cv_results.iloc[0]['best_params']}")
    print("\nPerformance:")
    print(f"  CV RMSE: {cv_results.iloc[0]['cv_rmse']:.2f}")
    print(f"  CV R²: {cv_results.iloc[0]['cv_r2']:.3f}")
    print(f"  Test RMSE: {evaluation['rmse']:.2f}")
    print(f"  Test R²: {evaluation['r2']:.3f}")
    print(f"  Test MAE: {evaluation['mae']:.2f}")
    print(f"  Test MAPE: {evaluation['mape']:.2f}%")
    
    # Print top features if available
    if 'feature_importance' in explanation:
        top_features = explanation['feature_importance'].head(5)
        print("\nTop 5 Important Features:")
        for i, (feature, importance) in enumerate(zip(top_features['feature'], top_features['importance']), 1):
            print(f"  {i}. {feature}: {importance:.4f}")
    
    print("\nModel files saved to:")
    print(f"  {os.path.join(MODEL_DIR, 'price_model.joblib')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'model_results')}")
    print("="*50)
