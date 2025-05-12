"""
Ridge Regression Cross-Validation Script

This script performs thorough cross-validation of Ridge Regression models
for computer price prediction, optimizing hyperparameters and evaluating performance.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, cross_validate
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
import joblib

# Add the src directory to the Python path
sys.path.append('./src')

# Import project modules
from features import engineer_features, identify_core_features, get_feature_names
from positive_price_scaler import PositivePriceScaler

def load_and_prepare_data(data_path='./data/db_computers_2025_processed.csv'):
    """Load and prepare the data for modeling."""
    print(f"\n=== Loading data from {data_path} ===")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nPrice statistics:")
    print(f"Mean: {df['Price'].mean():.2f} €")
    print(f"Std Dev: {df['Price'].std():.2f} €")
    print(f"Min: {df['Price'].min():.2f} €")
    print(f"Max: {df['Price'].max():.2f} €")
    
    # Create price scaler
    price_scaler = PositivePriceScaler(price_column='Price', min_price=100)
    price_scaler.fit(df)
    
    # Scale the target
    scaled_prices = price_scaler.transform_target(df['Price'])
    
    # Engineer features
    numeric_features, categorical_features = identify_core_features(df)
    X_transformed, feature_pipeline = engineer_features(
        df,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )
    
    # Get feature names
    feature_names = get_feature_names(feature_pipeline, numeric_features, categorical_features)
    
    return {
        'df': df,
        'X': X_transformed,
        'y': scaled_prices,
        'feature_names': feature_names,
        'price_scaler': price_scaler,
        'feature_pipeline': feature_pipeline
    }

def perform_ridge_cross_validation(X, y, price_scaler, alphas=None, cv_folds=5):
    """Perform cross-validation for Ridge Regression with multiple alphas."""
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    print(f"\n=== Ridge Regression Cross-Validation (k={cv_folds}) ===")
    print(f"Testing alpha values: {alphas}")
    
    # Create KFold object for consistent splits
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Store results
    results = {
        'alpha': [],
        'mean_cv_rmse': [],
        'std_cv_rmse': [],
        'mean_cv_r2': [],
        'std_cv_r2': [],
        'mean_cv_mae': [],
        'std_cv_mae': [],
        'mean_cv_rmse_orig': [],
        'std_cv_rmse_orig': [],
        'mean_cv_mae_orig': [],
        'std_cv_mae_orig': []
    }
    
    # Define a function to compute metrics in original price scale
    def calc_metrics_orig(y_true, y_pred):
        y_true_orig = price_scaler.inverse_transform(None, y_true)
        y_pred_orig = price_scaler.inverse_transform(None, y_pred)
        
        rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
        mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
        r2_orig = r2_score(y_true_orig, y_pred_orig)
        
        return rmse_orig, mae_orig, r2_orig
    
    # Perform CV for each alpha
    for alpha in alphas:
        print(f"\nEvaluating Ridge with alpha={alpha}")
        ridge = Ridge(alpha=alpha, random_state=42)
        
        # Standard metrics in scaled space
        cv_results = cross_validate(
            ridge, X, y, 
            cv=kf,
            scoring={
                'neg_mean_squared_error': 'neg_mean_squared_error',
                'r2': 'r2',
                'neg_mean_absolute_error': 'neg_mean_absolute_error'
            },
            return_train_score=True
        )
        
        # Calculate RMSE from MSE
        cv_rmse = np.sqrt(-cv_results['test_neg_mean_squared_error'])
        cv_mae = -cv_results['test_neg_mean_absolute_error']
        cv_r2 = cv_results['test_r2']
        
        # Calculate metrics in original price scale
        cv_rmse_orig = []
        cv_mae_orig = []
        cv_r2_orig = []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            rmse_orig, mae_orig, r2_orig = calc_metrics_orig(y_test, y_pred)
            cv_rmse_orig.append(rmse_orig)
            cv_mae_orig.append(mae_orig)
            cv_r2_orig.append(r2_orig)
        
        # Store the results
        results['alpha'].append(alpha)
        results['mean_cv_rmse'].append(cv_rmse.mean())
        results['std_cv_rmse'].append(cv_rmse.std())
        results['mean_cv_r2'].append(cv_r2.mean())
        results['std_cv_r2'].append(cv_r2.std())
        results['mean_cv_mae'].append(cv_mae.mean())
        results['std_cv_mae'].append(cv_mae.std())
        results['mean_cv_rmse_orig'].append(np.mean(cv_rmse_orig))
        results['std_cv_rmse_orig'].append(np.std(cv_rmse_orig))
        results['mean_cv_mae_orig'].append(np.mean(cv_mae_orig))
        results['std_cv_mae_orig'].append(np.std(cv_mae_orig))
        
        # Print the results
        print(f"Cross-Validation Results for alpha={alpha}:")
        print(f"  Scaled Space:")
        print(f"    RMSE: {cv_rmse.mean():.4f} (±{cv_rmse.std():.4f})")
        print(f"    MAE: {cv_mae.mean():.4f} (±{cv_mae.std():.4f})")
        print(f"    R²: {cv_r2.mean():.4f} (±{cv_r2.std():.4f})")
        print(f"  Original Price Space:")
        print(f"    RMSE: {np.mean(cv_rmse_orig):.2f} € (±{np.std(cv_rmse_orig):.2f} €)")
        print(f"    MAE: {np.mean(cv_mae_orig):.2f} € (±{np.std(cv_mae_orig):.2f} €)")
        print(f"    R²: {np.mean(cv_r2_orig):.4f} (±{np.std(cv_r2_orig):.4f})")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).set_index('alpha')
    
    return results_df

def plot_cv_results(results_df):
    """Plot the cross-validation results."""
    plt.figure(figsize=(14, 10))
    
    # Plot RMSE
    plt.subplot(2, 2, 1)
    plt.errorbar(
        results_df.index,
        results_df['mean_cv_rmse'],
        yerr=results_df['std_cv_rmse'],
        fmt='o-',
        capsize=5
    )
    plt.xscale('log')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('RMSE (Scaled Space)')
    plt.title('RMSE vs Alpha')
    plt.grid(True)
    
    # Plot R²
    plt.subplot(2, 2, 2)
    plt.errorbar(
        results_df.index,
        results_df['mean_cv_r2'],
        yerr=results_df['std_cv_r2'],
        fmt='o-',
        capsize=5
    )
    plt.xscale('log')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('R² (Scaled Space)')
    plt.title('R² vs Alpha')
    plt.grid(True)
    
    # Plot Original RMSE
    plt.subplot(2, 2, 3)
    plt.errorbar(
        results_df.index,
        results_df['mean_cv_rmse_orig'],
        yerr=results_df['std_cv_rmse_orig'],
        fmt='o-',
        capsize=5
    )
    plt.xscale('log')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('RMSE (Original Price)')
    plt.title('Original RMSE vs Alpha')
    plt.grid(True)
    
    # Plot Original MAE
    plt.subplot(2, 2, 4)
    plt.errorbar(
        results_df.index,
        results_df['mean_cv_mae_orig'],
        yerr=results_df['std_cv_mae_orig'],
        fmt='o-',
        capsize=5
    )
    plt.xscale('log')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('MAE (Original Price)')
    plt.title('Original MAE vs Alpha')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./outputs/ridge_cv_results.png')
    print(f"\nResults plot saved to ./outputs/ridge_cv_results.png")

def train_final_model(X, y, best_alpha, price_scaler, feature_pipeline):
    """Train the final Ridge model with the best alpha."""
    print(f"\n=== Training Final Ridge Model with alpha={best_alpha} ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = Ridge(alpha=best_alpha, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"Test Set Performance (Scaled Space):")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  R²: {test_r2:.4f}")
    
    # Calculate metrics in original space
    y_test_orig = price_scaler.inverse_transform(None, y_test)
    y_pred_orig = price_scaler.inverse_transform(None, y_pred)
    
    test_rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    test_mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
    test_r2_orig = r2_score(y_test_orig, y_pred_orig)
    
    print(f"Test Set Performance (Original Price Space):")
    print(f"  RMSE: {test_rmse_orig:.2f} €")
    print(f"  MAE: {test_mae_orig:.2f} €")
    print(f"  R²: {test_r2_orig:.4f}")
    
    # Save the model
    os.makedirs('./models', exist_ok=True)
    joblib.dump(model, './models/ridge_model.joblib')
    joblib.dump(price_scaler, './models/ridge_price_scaler.joblib')
    joblib.dump(feature_pipeline, './models/ridge_feature_pipeline.joblib')
    
    print("\nFinal model, price scaler, and feature pipeline saved to models directory.")
    
    return model

def main():
    """Main function to run the Ridge Regression cross-validation."""
    print("=== Ridge Regression Cross-Validation for Computer Price Prediction ===")
    
    # Create output directory
    os.makedirs('./outputs', exist_ok=True)
    
    # Load and prepare data
    data = load_and_prepare_data()
    
    # Define alpha values to test
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    # Perform cross-validation
    results_df = perform_ridge_cross_validation(
        data['X'], data['y'], data['price_scaler'], alphas, cv_folds=5
    )
    
    # Save results
    results_df.to_csv('./outputs/ridge_cv_results.csv')
    print(f"\nCross-validation results saved to ./outputs/ridge_cv_results.csv")
    
    # Plot results
    plot_cv_results(results_df)
    
    # Get the best alpha (based on RMSE)
    best_alpha = results_df['mean_cv_rmse'].idxmin()
    print(f"\nBest alpha based on RMSE: {best_alpha}")
    
    # Train final model
    train_final_model(
        data['X'], data['y'], best_alpha, 
        data['price_scaler'], data['feature_pipeline']
    )
    
    print("\n=== Ridge Regression Cross-Validation Complete ===")
    print("You can now use the trained Ridge model in the app.")

if __name__ == "__main__":
    main()
