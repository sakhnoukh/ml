#!/usr/bin/env python3
"""
Model Preparation Module for ML Marketplace.

This module performs final preparation of the engineered features before modeling:
1. Dataset exploration and visualization
2. Outlier detection and handling
3. Feature selection
4. Dimensionality reduction
5. Train/test/validation splitting
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.metrics import r2_score
import gc
from typing import List, Dict, Optional, Union, Tuple
import joblib

# Set up paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "model_prep")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_engineered_data(file_path=None):
    """
    Load the engineered feature dataset.
    
    Args:
        file_path: Path to engineered features file
        
    Returns:
        DataFrame with engineered features
    """
    if file_path is None:
        file_path = os.path.join(DATA_DIR, "db_computers_features.csv")
    
    return pd.read_csv(file_path, encoding='utf-8-sig')

def explore_dataset(df, target_col='Precio', save_plots=True):
    """
    Explore the engineered dataset and generate visualizations.
    
    Args:
        df: DataFrame with engineered features
        target_col: Name of target column
        save_plots: Whether to save plots to disk
        
    Returns:
        Summary statistics and insights
    """
    print(f"Dataset shape: {df.shape}")
    print(f"\nSample of data:")
    print(df.head())
    
    # Summary statistics
    print("\nSummary statistics:")
    stats = df.describe().T
    print(stats)
    
    # Save statistics to file
    stats.to_csv(os.path.join(OUTPUT_DIR, "engineered_features_stats.csv"))
    
    # Check for missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing_percent = (missing / len(df)) * 100
    missing_stats = pd.DataFrame({
        'missing_count': missing,
        'missing_percent': missing_percent
    })
    
    if not missing_stats.empty:
        print("\nMissing values:")
        print(missing_stats)
        missing_stats.to_csv(os.path.join(OUTPUT_DIR, "missing_values_after_engineering.csv"))
    else:
        print("\nNo missing values found!")
    
    # Check for constant or near-constant features
    variance = df.var()
    low_var_features = variance[variance < 0.01].index.tolist()
    if low_var_features:
        print(f"\nLow variance features (may be removed): {len(low_var_features)}")
        print(low_var_features[:10], "..." if len(low_var_features) > 10 else "")
    
    if save_plots:
        # Create plots directory
        plots_dir = os.path.join(OUTPUT_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Distribution of target variable
        if target_col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[target_col], kde=True)
            plt.title(f"Distribution of {target_col}")
            plt.savefig(os.path.join(plots_dir, "target_distribution.png"))
            plt.close()
            
            # Log transform for skewed target
            if df[target_col].skew() > 1:
                plt.figure(figsize=(10, 6))
                sns.histplot(np.log1p(df[target_col]), kde=True)
                plt.title(f"Distribution of log({target_col})")
                plt.savefig(os.path.join(plots_dir, "target_log_distribution.png"))
                plt.close()
        
        # Feature distributions (sample of numeric features)
        numeric_features = df.select_dtypes(include=['number']).columns.tolist()
        if target_col in numeric_features:
            numeric_features.remove(target_col)
        
        # Only plot a sample of feature distributions to avoid generating too many plots
        sample_features = np.random.choice(numeric_features, min(10, len(numeric_features)), replace=False)
        for feature in sample_features:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[feature].dropna(), kde=True)
            plt.title(f"Distribution of {feature}")
            plt.savefig(os.path.join(plots_dir, f"feature_dist_{feature.replace(' ', '_')}.png"))
            plt.close()
    
    return {
        'shape': df.shape,
        'missing': not missing_stats.empty,
        'low_var_features': low_var_features,
        'target_skew': df[target_col].skew() if target_col in df.columns else None
    }

def handle_outliers(df, target_col='Precio', method='winsorize', threshold=0.01):
    """
    Detect and handle outliers in the dataset.
    
    Args:
        df: DataFrame with engineered features
        target_col: Name of target column
        method: Method for handling outliers ('winsorize', 'clip', or 'remove')
        threshold: Threshold for percentile-based outlier detection
        
    Returns:
        DataFrame with outliers handled
    """
    print(f"Handling outliers using {method} method...")
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Get numeric columns except target
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Handle outliers for each numeric column
    outlier_info = {}
    
    for col in numeric_cols:
        # Calculate percentiles
        lower_bound = df[col].quantile(threshold)
        upper_bound = df[col].quantile(1 - threshold)
        
        # Count outliers
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        if outliers > 0:
            outlier_info[col] = {
                'count': outliers,
                'percentage': (outliers / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            # Handle outliers based on method
            if method == 'winsorize':
                df_clean[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == 'clip':
                df_clean[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))
            elif method == 'remove':
                mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                df_clean = df_clean[mask]
    
    # Print outlier summary
    total_outliers = sum(info['count'] for info in outlier_info.values())
    print(f"Found {total_outliers} outliers across {len(outlier_info)} features")
    print(f"Original shape: {df.shape}, After outlier handling: {df_clean.shape}")
    
    # Save outlier info
    if outlier_info:
        outlier_df = pd.DataFrame({
            'feature': list(outlier_info.keys()),
            'outlier_count': [info['count'] for info in outlier_info.values()],
            'outlier_percentage': [info['percentage'] for info in outlier_info.values()],
            'lower_bound': [info['lower_bound'] for info in outlier_info.values()],
            'upper_bound': [info['upper_bound'] for info in outlier_info.values()]
        })
        outlier_df.to_csv(os.path.join(OUTPUT_DIR, "outlier_info.csv"), index=False)
    
    return df_clean

def select_features(df, target_col='Precio', method='combined', n_features=None):
    """
    Select the most important features using multiple methods.
    
    Args:
        df: DataFrame with engineered features
        target_col: Name of target column
        method: Feature selection method ('rf', 'f_regression', or 'combined')
        n_features: Number of features to select (if None, automatic)
        
    Returns:
        DataFrame with selected features and list of selected feature names
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Set default number of features if not specified
    if n_features is None:
        n_features = min(50, X.shape[1] // 2)  # Use half of features by default, max 50
    
    # Dictionary to store feature importance by method
    feature_importance = {}
    
    # Method 1: Random Forest importance
    if method in ['rf', 'combined']:
        print(f"Using Random Forest for feature importance...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importance['rf'] = rf_importance
        
        # Save feature importance
        rf_importance.to_csv(os.path.join(OUTPUT_DIR, "rf_feature_importance.csv"), index=False)
        
        # Plot top features
        top_n = min(20, len(rf_importance))
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=rf_importance.head(top_n))
        plt.title(f"Top {top_n} Features by Random Forest Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", "rf_feature_importance.png"))
        plt.close()
    
    # Method 2: F-regression (linear correlation)
    if method in ['f_regression', 'combined']:
        print(f"Using F-regression for feature importance...")
        f_selector = SelectKBest(f_regression, k='all')
        f_selector.fit(X, y)
        f_importance = pd.DataFrame({
            'feature': X.columns,
            'f_score': f_selector.scores_,
            'p_value': f_selector.pvalues_
        }).sort_values('f_score', ascending=False)
        feature_importance['f_regression'] = f_importance
        
        # Save feature importance
        f_importance.to_csv(os.path.join(OUTPUT_DIR, "f_regression_importance.csv"), index=False)
        
        # Plot top features
        top_n = min(20, len(f_importance))
        plt.figure(figsize=(12, 8))
        sns.barplot(x='f_score', y='feature', data=f_importance.head(top_n))
        plt.title(f"Top {top_n} Features by F-regression Score")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", "f_regression_importance.png"))
        plt.close()
    
    # Combined approach
    if method == 'combined':
        # Combine importance ranks
        rf_ranks = pd.Series(range(1, len(X.columns) + 1), 
                           index=feature_importance['rf']['feature']).to_dict()
        f_ranks = pd.Series(range(1, len(X.columns) + 1), 
                          index=feature_importance['f_regression']['feature']).to_dict()
        
        # Calculate combined rank
        combined_ranks = {}
        for feature in X.columns:
            combined_ranks[feature] = (rf_ranks.get(feature, len(X.columns)) + 
                                     f_ranks.get(feature, len(X.columns))) / 2
        
        combined_importance = pd.DataFrame({
            'feature': list(combined_ranks.keys()),
            'combined_rank': list(combined_ranks.values())
        }).sort_values('combined_rank')
        
        # Save combined importance
        combined_importance.to_csv(os.path.join(OUTPUT_DIR, "combined_feature_importance.csv"), index=False)
        
        # Select top features
        selected_features = combined_importance['feature'].iloc[:n_features].tolist()
    else:
        # Select based on individual method
        if method == 'rf':
            selected_features = feature_importance['rf']['feature'].iloc[:n_features].tolist()
        else:  # f_regression
            selected_features = feature_importance['f_regression']['feature'].iloc[:n_features].tolist()
    
    # Add target back
    selected_features.append(target_col)
    
    # Create dataset with selected features
    df_selected = df[selected_features].copy()
    
    print(f"Selected {len(selected_features) - 1} features out of {X.shape[1]} original features")
    print(f"Shape after feature selection: {df_selected.shape}")
    
    # Save list of selected features
    pd.Series(selected_features).to_csv(os.path.join(OUTPUT_DIR, "selected_features.csv"), index=False, header=['feature'])
    
    return df_selected, selected_features

def reduce_dimensions(df, target_col='Precio', method='pca', n_components=None, variance_threshold=0.95):
    """
    Reduce dimensionality of the feature set.
    
    Args:
        df: DataFrame with features
        target_col: Name of target column
        method: Dimensionality reduction method (only 'pca' supported for now)
        n_components: Number of components to keep (if None, use variance_threshold)
        variance_threshold: Amount of variance to retain if n_components is None
        
    Returns:
        DataFrame with reduced dimensions and PCA transformer
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    
    # Get number of components
    if n_components is None:
        n_components = min(X.shape[1], round(X.shape[1] * 0.5))  # Default to 50% of features
    
    print(f"Reducing dimensions using {method} from {X.shape[1]} to {n_components}...")
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Create DataFrame with transformed data
    pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
    
    # Add target column back
    df_pca[target_col] = df[target_col].values
    
    # Print variance explained
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"Total variance explained: {cumulative_variance[-1]:.4f}")
    print(f"First 5 components explain: {cumulative_variance[min(4, len(cumulative_variance)-1)]:.4f} of variance")
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by PCA Components')
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% Threshold')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95% Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "pca_explained_variance.png"))
    plt.close()
    
    # Save PCA transformer
    joblib.dump(pca, os.path.join(MODEL_DIR, "pca_transformer.joblib"))
    
    # Save component information
    component_info = pd.DataFrame({
        'component': pca_cols,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance
    })
    component_info.to_csv(os.path.join(OUTPUT_DIR, "pca_components_info.csv"), index=False)
    
    # Save feature loadings
    feature_loadings = pd.DataFrame(
        pca.components_.T, 
        columns=pca_cols,
        index=X.columns
    )
    feature_loadings.to_csv(os.path.join(OUTPUT_DIR, "pca_feature_loadings.csv"))
    
    return df_pca, pca

def create_data_splits(df, target_col='Precio', test_size=0.2, valid_size=0.1, random_state=42):
    """
    Create train, validation, and test splits.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        test_size: Proportion of data for testing
        valid_size: Proportion of data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with train, validation, and test DataFrames
    """
    # Calculate effective validation size relative to remaining data after test split
    # If test_size is 0.2 and valid_size is 0.1, then valid split is 0.1/(1-0.2) = 0.125 of remaining data
    effective_valid_size = valid_size / (1 - test_size)
    
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # First split: separate test set
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate train and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=effective_valid_size, random_state=random_state
    )
    
    # Create full DataFrames with target
    train_df = pd.concat([X_train, y_train], axis=1)
    valid_df = pd.concat([X_valid, y_valid], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Print split sizes
    print(f"Data splits created:")
    print(f"  Train set: {train_df.shape[0]} samples ({train_df.shape[0]/df.shape[0]:.1%})")
    print(f"  Validation set: {valid_df.shape[0]} samples ({valid_df.shape[0]/df.shape[0]:.1%})")
    print(f"  Test set: {test_df.shape[0]} samples ({test_df.shape[0]/df.shape[0]:.1%})")
    
    # Save splits
    train_df.to_csv(os.path.join(DATA_DIR, "train_set.csv"), index=False)
    valid_df.to_csv(os.path.join(DATA_DIR, "validation_set.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test_set.csv"), index=False)
    
    return {
        'train': train_df,
        'validation': valid_df,
        'test': test_df
    }

def prepare_for_modeling(input_file=None, target_col='Precio'):
    """
    Main function to prepare data for modeling.
    
    Args:
        input_file: Path to engineered features file
        target_col: Name of target column
        
    Returns:
        Dictionary with train, validation, and test data
    """
    print("Starting model preparation...")
    
    # Load engineered data
    df = load_engineered_data(input_file)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Explore dataset
    print("\n===== 1. Data Exploration =====")
    explore_results = explore_dataset(df, target_col=target_col)
    
    # Handle outliers
    print("\n===== 2. Outlier Handling =====")
    df_clean = handle_outliers(df, target_col=target_col, method='winsorize')
    
    # Select features
    print("\n===== 3. Feature Selection =====")
    df_selected, selected_features = select_features(df_clean, target_col=target_col, method='combined')
    
    # Reduce dimensions
    print("\n===== 4. Dimensionality Reduction =====")
    # Run PCA on the data with selected features, keeping 90% of variance
    df_pca, pca = reduce_dimensions(df_selected, target_col=target_col, 
                                  n_components=min(20, len(selected_features) - 1))  # -1 for the target
    
    # Create data splits (use the dimensionality-reduced version)
    print("\n===== 5. Data Splitting =====")
    splits = create_data_splits(df_pca, target_col=target_col)
    
    print("\nModel preparation complete!")
    return splits

if __name__ == "__main__":
    # Run full preparation pipeline
    data_splits = prepare_for_modeling()
    
    # Print final message
    print("\nData is ready for modeling!")
    print(f"Train, validation, and test sets saved to {DATA_DIR}")
    print(f"Preparation outputs saved to {OUTPUT_DIR}")
    print(f"PCA transformer saved to {MODEL_DIR}")
