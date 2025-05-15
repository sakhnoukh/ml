#!/usr/bin/env python3
"""
ML Marketplace Model Optimizer

This script optimizes the price prediction model through:
1. Advanced feature engineering
2. Hyperparameter tuning with RandomizedSearchCV
3. Model selection and ensembling
4. Cross-validation with stratification
5. Learning curve analysis

The optimized model is saved for later use in the web UI.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, RandomizedSearchCV, 
                                   cross_val_score, learning_curve, KFold)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                            VotingRegressor, StackingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from scipy.stats import randint, uniform
import joblib
import logging
import gc
import time
from typing import Dict, List, Tuple, Any
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        "logs", "model_optimization.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "optimized_model")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def load_and_preprocess_data(train_path=None, test_path=None, target_col='Precio_Rango'):
    """
    Load and preprocess data for optimization.
    
    Args:
        train_path: Path to training data file
        test_path: Path to test data file
        target_col: Target column for prediction
        
    Returns:
        Dictionary with preprocessed data splits
    """
    # Default paths
    if train_path is None:
        train_path = os.path.join(DATA_DIR, "train_set.csv")
    if test_path is None:
        test_path = os.path.join(DATA_DIR, "test_set.csv")
    
    logger.info(f"Loading data from {train_path} and {test_path}")
    
    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Separate features and target
    X_train = train_data.drop(columns=[target_col]) if target_col in train_data.columns else train_data
    y_train = train_data[target_col] if target_col in train_data.columns else None
    
    X_test = test_data.drop(columns=[target_col]) if target_col in test_data.columns else test_data
    y_test = test_data[target_col] if target_col in test_data.columns else None
    
    logger.info(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Get numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()
    
    logger.info(f"Found {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features")
    
    # Clean numeric data
    for col in numeric_cols:
        # Handle any remaining numeric issues
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
        
        # Fill missing values
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
    
    # For now, we'll use only numeric features for optimization
    # We'll handle categorical features in the separate feature_engineering.py file
    X_train_numeric = X_train[numeric_cols]
    X_test_numeric = X_test[numeric_cols]
    
    logger.info(f"Preprocessed data: X_train shape: {X_train_numeric.shape}, X_test shape: {X_test_numeric.shape}")
    
    return {
        'X_train': X_train,
        'X_train_numeric': X_train_numeric,
        'y_train': y_train,
        'X_test': X_test,
        'X_test_numeric': X_test_numeric,
        'y_test': y_test,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }

def optimize_random_forest(X_train, y_train, cv=5, n_iter=20, random_state=42):
    """
    Optimize Random Forest model using RandomizedSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        n_iter: Number of parameter settings to try
        random_state: Random seed
        
    Returns:
        Optimized model and results
    """
    logger.info("Optimizing Random Forest model...")
    
    # Parameter grid
    param_grid = {
        'n_estimators': randint(100, 500),
        'max_depth': [None] + list(randint(5, 50).rvs(5, random_state=random_state)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None] + list(uniform(0.1, 0.9).rvs(3, random_state=random_state))
    }
    
    # Create base model
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    
    # RandomizedSearchCV
    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )
    
    # Fit model
    start_time = time.time()
    rf_search.fit(X_train, y_train)
    end_time = time.time()
    
    # Log results
    logger.info(f"Random Forest optimization completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Best parameters: {rf_search.best_params_}")
    logger.info(f"Best RMSE: {-rf_search.best_score_:.4f}")
    
    # Get best model
    best_rf = rf_search.best_estimator_
    
    # Create results summary
    results = pd.DataFrame(rf_search.cv_results_)
    results = results.sort_values('rank_test_score')
    
    # Save results
    results.to_csv(os.path.join(OUTPUT_DIR, "rf_optimization_results.csv"), index=False)
    
    return best_rf, results

def optimize_gradient_boosting(X_train, y_train, cv=5, n_iter=20, random_state=42):
    """
    Optimize Gradient Boosting model using RandomizedSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        n_iter: Number of parameter settings to try
        random_state: Random seed
        
    Returns:
        Optimized model and results
    """
    logger.info("Optimizing Gradient Boosting model...")
    
    # Parameter grid
    param_grid = {
        'n_estimators': randint(50, 300),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'subsample': uniform(0.6, 0.4),
        'max_features': uniform(0.1, 0.9)
    }
    
    # Create base model
    gb = GradientBoostingRegressor(random_state=random_state)
    
    # RandomizedSearchCV
    gb_search = RandomizedSearchCV(
        estimator=gb,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )
    
    # Fit model
    start_time = time.time()
    gb_search.fit(X_train, y_train)
    end_time = time.time()
    
    # Log results
    logger.info(f"Gradient Boosting optimization completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Best parameters: {gb_search.best_params_}")
    logger.info(f"Best RMSE: {-gb_search.best_score_:.4f}")
    
    # Get best model
    best_gb = gb_search.best_estimator_
    
    # Create results summary
    results = pd.DataFrame(gb_search.cv_results_)
    results = results.sort_values('rank_test_score')
    
    # Save results
    results.to_csv(os.path.join(OUTPUT_DIR, "gb_optimization_results.csv"), index=False)
    
    return best_gb, results

def build_ensemble_model(X_train, y_train, X_test, y_test, base_models, cv=5):
    """
    Build and evaluate ensemble models.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        base_models: List of tuples with (name, model)
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary with ensemble models and evaluation results
    """
    logger.info("Building ensemble models...")
    
    # Voting Regressor
    voting_reg = VotingRegressor(estimators=base_models)
    
    # Stacking Regressor with Ridge as final estimator
    stacking_reg = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(),
        cv=KFold(n_splits=cv, shuffle=True, random_state=42)
    )
    
    # Train models
    logger.info("Training Voting Regressor...")
    voting_reg.fit(X_train, y_train)
    
    logger.info("Training Stacking Regressor...")
    stacking_reg.fit(X_train, y_train)
    
    # Evaluate models
    models = {
        'voting': voting_reg,
        'stacking': stacking_reg
    }
    
    results = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        logger.info(f"{name.capitalize()} Ensemble Results:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        
        results[name] = {
            'model': model,
            'rmse': rmse,
            'r2': r2,
            'mae': mae
        }
    
    # Save models
    for name, result in results.items():
        model_path = os.path.join(MODEL_DIR, f"{name}_ensemble_model.joblib")
        joblib.dump(result['model'], model_path)
        logger.info(f"Saved {name} model to {model_path}")
    
    return results

def evaluate_and_compare_models(models_dict, X_test, y_test):
    """
    Evaluate and compare multiple models.
    
    Args:
        models_dict: Dictionary of models to evaluate
        X_test: Test features
        y_test: Test target
        
    Returns:
        DataFrame with comparison results
    """
    logger.info("Evaluating and comparing models...")
    
    results = []
    
    for name, model in models_dict.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Add to results
        results.append({
            'model': name,
            'rmse': rmse,
            'r2': r2,
            'mae': mae
        })
    
    # Create DataFrame
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('rmse')
    
    # Save results
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    
    # Log best model
    best_model = comparison_df.iloc[0]
    logger.info(f"Best model: {best_model['model']}")
    logger.info(f"  RMSE: {best_model['rmse']:.4f}")
    logger.info(f"  R²: {best_model['r2']:.4f}")
    logger.info(f"  MAE: {best_model['mae']:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # RMSE
    plt.subplot(1, 3, 1)
    sns.barplot(x='model', y='rmse', data=comparison_df)
    plt.title('RMSE (lower is better)')
    plt.xticks(rotation=45)
    
    # R²
    plt.subplot(1, 3, 2)
    sns.barplot(x='model', y='r2', data=comparison_df)
    plt.title('R² (higher is better)')
    plt.xticks(rotation=45)
    
    # MAE
    plt.subplot(1, 3, 3)
    sns.barplot(x='model', y='mae', data=comparison_df)
    plt.title('MAE (lower is better)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"))
    plt.close()
    
    return comparison_df

def analyze_learning_curve(model, X_train, y_train, cv=5, n_jobs=-1):
    """
    Analyze learning curve to detect overfitting/underfitting.
    
    Args:
        model: Model to analyze
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        n_jobs: Number of jobs for parallel processing
        
    Returns:
        None (saves plot to file)
    """
    logger.info("Analyzing learning curve...")
    
    # Get model name
    model_name = type(model).__name__
    
    # Set up train sizes
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, 
        train_sizes=train_sizes, 
        cv=cv, 
        scoring='neg_root_mean_squared_error',
        n_jobs=n_jobs
    )
    
    # Calculate statistics
    train_mean = -np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = -np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training RMSE')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation RMSE')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    plt.xlabel('Training set size')
    plt.ylabel('RMSE')
    plt.title(f'Learning Curve for {model_name}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_learning_curve.png"))
    plt.close()
    
    # Log analysis
    gap = test_mean[-1] - train_mean[-1]
    logger.info(f"Learning curve analysis for {model_name}:")
    logger.info(f"  Final training RMSE: {train_mean[-1]:.4f}")
    logger.info(f"  Final validation RMSE: {test_mean[-1]:.4f}")
    logger.info(f"  Gap (validation - training): {gap:.4f}")
    
    if gap > 0.1:
        logger.info("  Model might be overfitting (large gap between training and validation)")
    elif train_mean[-1] > 0.5:
        logger.info("  Model might be underfitting (high training error)")
    else:
        logger.info("  Model seems well-balanced")

def save_final_model(model, feature_names=None, scaler=None):
    """
    Save final model along with preprocessing components.
    
    Args:
        model: Final model to save
        feature_names: List of feature names
        scaler: Scaler used for preprocessing
        
    Returns:
        Path to saved model
    """
    # Get model name
    model_name = type(model).__name__
    
    # Create model package
    model_package = {
        'model': model,
        'feature_names': feature_names,
        'scaler': scaler,
        'creation_date': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save model package
    model_path = os.path.join(MODEL_DIR, "optimized_model.joblib")
    joblib.dump(model_package, model_path)
    logger.info(f"Final optimized model saved to {model_path}")
    
    return model_path

if __name__ == "__main__":
    # Load and preprocess data
    logger.info("Starting model optimization process...")
    data = load_and_preprocess_data()
    
    # Split data
    X_train = data['X_train_numeric']
    y_train = data['y_train']
    X_test = data['X_test_numeric']
    y_test = data['y_test']
    
    # Optimize Random Forest
    best_rf, rf_results = optimize_random_forest(X_train, y_train, n_iter=20)
    
    # Optimize Gradient Boosting
    best_gb, gb_results = optimize_gradient_boosting(X_train, y_train, n_iter=20)
    
    # Create elastic net and ridge models (simpler models for ensemble diversity)
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    elastic.fit(X_train, y_train)
    
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    
    # Build ensemble models
    base_models = [
        ('rf', best_rf),
        ('gb', best_gb),
        ('elastic', elastic),
        ('ridge', ridge)
    ]
    
    ensemble_results = build_ensemble_model(X_train, y_train, X_test, y_test, base_models)
    
    # Compare all models
    models_to_compare = {
        'random_forest': best_rf,
        'gradient_boosting': best_gb,
        'elastic_net': elastic,
        'ridge': ridge,
        'voting_ensemble': ensemble_results['voting']['model'],
        'stacking_ensemble': ensemble_results['stacking']['model']
    }
    
    comparison = evaluate_and_compare_models(models_to_compare, X_test, y_test)
    
    # Get best model
    best_model_name = comparison.iloc[0]['model']
    best_model = models_to_compare[best_model_name]
    
    # Analyze learning curve for best model
    analyze_learning_curve(best_model, X_train, y_train)
    
    # Save final model
    save_final_model(best_model, feature_names=data['numeric_cols'])
    
    logger.info("Model optimization process completed successfully!")
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"  RMSE: {comparison.iloc[0]['rmse']:.4f}")
    logger.info(f"  R²: {comparison.iloc[0]['r2']:.4f}")
    logger.info(f"  MAE: {comparison.iloc[0]['mae']:.4f}")
