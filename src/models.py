"""
Predictive modeling module for the ML Marketplace.

This module handles training, evaluation, and interpretation of predictive models
for computer price prediction.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import shap
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_models(X: np.ndarray, y: np.ndarray, 
                feature_names: List[str] = None,
                cv_folds: int = 5) -> Dict[str, Dict[str, Any]]:
    """
    Train multiple regression models and evaluate with cross-validation.
    
    Args:
        X: Feature matrix
        y: Target values
        feature_names: Names of features (for feature importance)
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dictionary of model results
    """
    logger.info("Training multiple regression models with cross-validation")
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define more stable models to train with simpler parameters
    models = {
        'random_forest': {
            'model': RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=5),
            'param_grid': {
                'n_estimators': [100],
                'max_depth': [8, 15],
                'min_samples_split': [5]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.05, max_depth=5),
            'param_grid': {
                'n_estimators': [100],
                'learning_rate': [0.05],
                'max_depth': [5]
            }
        },
        # Linear model for simplicity and stability
        'linear_regression': {
            'model': Pipeline([
                ('linear', Ridge(alpha=1.0))
            ]),
            'param_grid': {
                'linear__alpha': [0.1, 1.0, 10.0]
            }
        }
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model_info in models.items():
        logger.info(f"Training {name} model")
        
        # Grid search for best parameters
        grid_search = GridSearchCV(
            model_info['model'],
            model_info['param_grid'],
            cv=cv_folds,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Cross-validation
        cv_scores = cross_val_score(
            best_model, X_train, y_train, 
            cv=cv_folds, 
            scoring='neg_root_mean_squared_error'
        )
        
        # Validate on holdout set
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        logger.info(f"{name} - CV RMSE: {-np.mean(cv_scores):.2f}, "
                   f"Validation RMSE: {rmse:.2f}, "
                   f"Validation MAE: {mae:.2f}, "
                   f"Validation R²: {r2:.2f}")
        
        # Calculate feature importances if available
        feature_importance = None
        if hasattr(best_model, 'feature_importances_') and feature_names is not None:
            feature_importance = dict(zip(feature_names, best_model.feature_importances_))
        
        # Store results
        results[name] = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'cv_rmse': -np.mean(cv_scores),
            'validation_rmse': rmse,
            'validation_mae': mae,
            'validation_r2': r2,
            'feature_importance': feature_importance
        }
    
    return results


def select_best_model(results: Dict[str, Dict[str, Any]]) -> Tuple[str, BaseEstimator]:
    """
    Select the best model based on cross-validation RMSE.
    
    Args:
        results: Dictionary of model results
        
    Returns:
        Tuple of (best model name, best model)
    """
    logger.info("Selecting best model based on cross-validation RMSE")
    
    # Find model with lowest RMSE
    best_model_name = min(results, key=lambda x: results[x]['cv_rmse'])
    best_model = results[best_model_name]['model']
    
    logger.info(f"Best model: {best_model_name} with "
               f"CV RMSE: {results[best_model_name]['cv_rmse']:.2f}, "
               f"Validation RMSE: {results[best_model_name]['validation_rmse']:.2f}")
    
    return best_model_name, best_model


def compute_shap_values(model: BaseEstimator, X: np.ndarray, 
                        feature_names: List[str] = None) -> Tuple[shap.Explanation, List[str]]:
    """
    Compute SHAP values for feature interpretation.
    
    Args:
        model: Trained model
        X: Feature matrix
        feature_names: Names of features
        
    Returns:
        Tuple of (SHAP explanation object, feature names)
    """
    logger.info("Computing SHAP values for feature interpretation")
    
    # Create explainer based on model type
    if isinstance(model, RandomForestRegressor):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, GradientBoostingRegressor):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, MLPRegressor):
        explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
    else:
        explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
    
    # Calculate SHAP values
    shap_values = explainer(X[:100])  # Limit to 100 samples for efficiency
    
    return shap_values, feature_names


def create_shap_summary_plot(shap_values: shap.Explanation, feature_names: List[str],
                            output_path: Optional[str] = None):
    """
    Create SHAP summary plot for feature importance.
    
    Args:
        shap_values: SHAP explanation object
        feature_names: Names of features
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"SHAP summary plot saved to {output_path}")
    
    plt.close()


def calculate_shap_values_for_instance(model: BaseEstimator, X: np.ndarray, 
                                      instance: np.ndarray,
                                      feature_names: List[str] = None) -> Dict[str, float]:
    """
    Calculate SHAP values for a single instance.
    
    Args:
        model: Trained model
        X: Feature matrix (for background distribution)
        instance: Single instance to explain
        feature_names: Names of features
        
    Returns:
        Dictionary mapping feature names to SHAP values
    """
    # Create explainer based on model type
    if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
    
    # Calculate SHAP values for the instance
    shap_values = explainer(instance.reshape(1, -1))
    
    # Map SHAP values to feature names
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(instance.shape[0])]
    
    return dict(zip(feature_names, shap_values.values[0]))


def save_model(model: BaseEstimator, output_path: str):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        output_path: Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Saving model to {output_path}")
    joblib.dump(model, output_path)


def load_model(input_path: str) -> BaseEstimator:
    """
    Load trained model from disk.
    
    Args:
        input_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {input_path}")
    return joblib.load(input_path)


def train_and_save_model(X: np.ndarray, y: np.ndarray, 
                        feature_names: List[str] = None,
                        model_path: str = "../models/price_model.joblib",
                        shap_plot_path: str = "../models/shap_summary.png") -> Tuple[BaseEstimator, Dict[str, Dict[str, Any]]]:
    """
    Train models, select best one, and save it.
    
    Args:
        X: Feature matrix
        y: Target values
        feature_names: Names of features
        model_path: Path to save the best model
        shap_plot_path: Path to save the SHAP summary plot
        
    Returns:
        Tuple of (best model, all model results)
    """
    # Train multiple models
    results = train_models(X, y, feature_names)
    
    # Select best model
    best_model_name, best_model = select_best_model(results)
    
    # Compute SHAP values
    shap_values, _ = compute_shap_values(best_model, X, feature_names)
    
    # Create SHAP summary plot
    create_shap_summary_plot(shap_values, feature_names, shap_plot_path)
    
    # Save best model
    save_model(best_model, model_path)
    
    return best_model, results


if __name__ == "__main__":
    # Example usage
    # from preprocessing import preprocess_data
    # from features import engineer_features, get_feature_names
    # 
    # # Preprocess data
    # df = preprocess_data("../data/db_computers_2025_raw.csv")
    # 
    # # Engineer features
    # X_transformed, pipeline = engineer_features(df)
    # 
    # # Get feature names
    # numeric_features, categorical_features = identify_core_features(df)
    # feature_names = get_feature_names(pipeline, numeric_features, categorical_features)
    # 
    # # Train and save model
    # best_model, results = train_and_save_model(
    #     X_transformed, 
    #     df['Price'],
    #     feature_names,
    #     model_path="../models/price_model.joblib",
    #     shap_plot_path="../models/shap_summary.png"
    # )
    pass
