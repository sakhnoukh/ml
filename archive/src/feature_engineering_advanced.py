#!/usr/bin/env python3
"""
Advanced Feature Engineering for ML Marketplace

This script provides advanced feature engineering techniques to improve model performance:
1. Feature interaction discovery
2. Feature scaling and transformation
3. Hardware-specific feature creation
4. Anomaly detection and handling
5. Feature selection using mutual information
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib
import logging
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        "logs", "feature_engineering.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "feature_engineering")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(train_path=None, test_path=None, target_col='Precio_Rango'):
    """
    Load train and test data.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        target_col: Target column name
        
    Returns:
        Dictionary with train and test data
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
    
    # Get numeric columns only for now
    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    X_train_numeric = X_train[numeric_cols]
    X_test_numeric = X_test[numeric_cols]
    
    logger.info(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    logger.info(f"Numeric features: {len(numeric_cols)}")
    
    return {
        'X_train': X_train,
        'X_train_numeric': X_train_numeric,
        'y_train': y_train,
        'X_test': X_test,
        'X_test_numeric': X_test_numeric,
        'y_test': y_test,
        'numeric_cols': numeric_cols
    }

def detect_and_handle_outliers(X_train, X_test, contamination=0.05):
    """
    Detect and handle outliers using Isolation Forest.
    
    Args:
        X_train: Training features
        X_test: Test features
        contamination: Expected proportion of outliers
        
    Returns:
        Cleaned X_train and X_test
    """
    logger.info("Detecting and handling outliers...")
    
    # Create a copy of the data
    X_train_clean = X_train.copy()
    
    # Initialize Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit and predict outliers
    outliers = iso_forest.fit_predict(X_train)
    
    # Count outliers
    outlier_count = (outliers == -1).sum()
    logger.info(f"Detected {outlier_count} outliers ({outlier_count/len(X_train)*100:.2f}%)")
    
    # Remove outliers
    X_train_clean = X_train.iloc[outliers == 1]
    y_train_clean = y_train.iloc[outliers == 1] if y_train is not None else None
    
    # Save outlier detection model
    joblib.dump(iso_forest, os.path.join(OUTPUT_DIR, "outlier_detector.joblib"))
    
    return X_train_clean, X_test, y_train_clean

def create_polynomial_features(X_train, X_test, degree=2, interaction_only=True):
    """
    Create polynomial and interaction features.
    
    Args:
        X_train: Training features
        X_test: Test features
        degree: Polynomial degree
        interaction_only: Whether to include only interaction terms
        
    Returns:
        DataFrames with polynomial features
    """
    logger.info(f"Creating polynomial features (degree={degree}, interaction_only={interaction_only})...")
    
    # Initialize polynomial features transformer
    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=False
    )
    
    # Fit and transform
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Get feature names
    feature_names = poly.get_feature_names_out(X_train.columns)
    
    # Create DataFrames
    X_train_poly_df = pd.DataFrame(X_train_poly, columns=feature_names)
    X_test_poly_df = pd.DataFrame(X_test_poly, columns=feature_names)
    
    logger.info(f"Created {X_train_poly_df.shape[1]} polynomial features")
    
    # Save transformer
    joblib.dump(poly, os.path.join(OUTPUT_DIR, "polynomial_transformer.joblib"))
    
    return X_train_poly_df, X_test_poly_df

def engineer_hardware_features(X_train, X_test):
    """
    Create specialized hardware-related features.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        DataFrames with hardware features
    """
    logger.info("Engineering hardware-specific features...")
    
    # Create copies
    X_train_hw = X_train.copy()
    X_test_hw = X_test.copy()
    
    # Find hardware features
    hardware_cols = {
        'storage': [col for col in X_train.columns if any(term in col.lower() 
                                                     for term in ['ssd', 'disco', 'storage', 'capacity'])],
        'screen': [col for col in X_train.columns if any(term in col.lower() 
                                                    for term in ['screen', 'pantalla', 'display'])],
        'processor': [col for col in X_train.columns if any(term in col.lower() 
                                                       for term in ['processor', 'procesador', 'cpu'])],
        'ram': [col for col in X_train.columns if any(term in col.lower() 
                                                for term in ['ram', 'memoria'])],
        'gpu': [col for col in X_train.columns if any(term in col.lower() 
                                                for term in ['gpu', 'graphic', 'gráfica'])]
    }
    
    # Log found features
    for category, cols in hardware_cols.items():
        if cols:
            logger.info(f"Found {len(cols)} {category} features")
    
    # Create combinations of hardware features
    for category, cols in hardware_cols.items():
        # Only use numeric columns
        numeric_cols = [col for col in cols if col in X_train.select_dtypes(include=['number']).columns]
        
        if len(numeric_cols) >= 2:
            logger.info(f"Creating feature combinations for {category}")
            
            # Create basic combinations
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    # Ratio feature (if values are > 0)
                    ratio_name = f"{col1}_to_{col2}_ratio"
                    X_train_hw[ratio_name] = X_train[col1] / (X_train[col2] + 1e-8)
                    X_test_hw[ratio_name] = X_test[col1] / (X_test[col2] + 1e-8)
                    
                    # Replace inf values
                    X_train_hw[ratio_name].replace([np.inf, -np.inf], np.nan, inplace=True)
                    X_test_hw[ratio_name].replace([np.inf, -np.inf], np.nan, inplace=True)
                    
                    # Fill NaN values
                    X_train_hw[ratio_name].fillna(X_train_hw[ratio_name].median(), inplace=True)
                    X_test_hw[ratio_name].fillna(X_train_hw[ratio_name].median(), inplace=True)
    
    # Create additional hardware features
    # 1. Performance-to-price ratio proxies
    if 'RAM_Memoria RAM' in X_train.columns and 'y_train' in locals():
        X_train_hw['ram_price_ratio'] = X_train['RAM_Memoria RAM'] / (y_train + 1e-8)
        # We don't create this for test set as we don't have y_test during prediction
    
    # 2. Storage and RAM combined
    if 'RAM_Memoria RAM' in X_train.columns and 'Disco duro_Capacidad de memoria SSD' in X_train.columns:
        X_train_hw['ram_ssd_sum'] = X_train['RAM_Memoria RAM'] + X_train['Disco duro_Capacidad de memoria SSD']
        X_test_hw['ram_ssd_sum'] = X_test['RAM_Memoria RAM'] + X_test['Disco duro_Capacidad de memoria SSD']
    
    # Count new features
    new_features = [col for col in X_train_hw.columns if col not in X_train.columns]
    logger.info(f"Created {len(new_features)} new hardware-related features")
    
    return X_train_hw, X_test_hw

def select_features(X_train, X_test, y_train, k=20):
    """
    Select top k features using mutual information.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        k: Number of features to select
        
    Returns:
        DataFrames with selected features and importance scores
    """
    logger.info(f"Selecting top {k} features using mutual information...")
    
    # Calculate mutual information
    mi_scores = mutual_info_regression(X_train, y_train)
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': mi_scores
    }).sort_values('importance', ascending=False)
    
    # Select top k features
    top_features = feature_importance.head(k)['feature'].tolist()
    
    logger.info(f"Top 5 features: {', '.join(top_features[:5])}")
    
    # Save feature importance
    feature_importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_mi.csv"), index=False)
    
    # Create DataFrames with selected features
    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]
    
    return X_train_selected, X_test_selected, feature_importance

def transform_features(X_train, X_test):
    """
    Apply various transformations to features.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        DataFrames with transformed features
    """
    logger.info("Applying feature transformations...")
    
    # Create copies
    X_train_trans = X_train.copy()
    X_test_trans = X_test.copy()
    
    # Apply log transformation to heavily skewed features
    for col in X_train.columns:
        # Check if column is numeric and has only positive values
        if pd.api.types.is_numeric_dtype(X_train[col]) and (X_train[col] > 0).all():
            # Calculate skewness
            skewness = X_train[col].skew()
            
            # Apply log transformation to heavily skewed features
            if abs(skewness) > 1:
                logger.info(f"Applying log transformation to {col} (skewness={skewness:.2f})")
                X_train_trans[f"{col}_log"] = np.log1p(X_train[col])
                X_test_trans[f"{col}_log"] = np.log1p(X_test[col])
    
    # Apply Yeo-Johnson transformation (doesn't require positive values)
    pt = PowerTransformer(method='yeo-johnson')
    
    # Select numeric columns only
    num_cols = X_train.select_dtypes(include=['number']).columns
    
    if len(num_cols) > 0:
        # Fit and transform
        X_train_pt = pd.DataFrame(
            pt.fit_transform(X_train[num_cols]),
            columns=[f"{col}_pt" for col in num_cols]
        )
        X_test_pt = pd.DataFrame(
            pt.transform(X_test[num_cols]),
            columns=[f"{col}_pt" for col in num_cols]
        )
        
        # Add to original DataFrames
        X_train_trans = pd.concat([X_train_trans, X_train_pt], axis=1)
        X_test_trans = pd.concat([X_test_trans, X_test_pt], axis=1)
        
        logger.info(f"Applied Yeo-Johnson transformation to {len(num_cols)} numeric features")
        
        # Save transformer
        joblib.dump(pt, os.path.join(OUTPUT_DIR, "power_transformer.joblib"))
    
    return X_train_trans, X_test_trans

def create_enhanced_dataset(X_train, X_test, y_train=None, y_test=None):
    """
    Create an enhanced dataset with all feature engineering techniques.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        
    Returns:
        Dictionary with enhanced datasets
    """
    logger.info("Creating enhanced dataset...")
    
    # Step 1: Detect and handle outliers (if y_train is available)
    if y_train is not None:
        X_train_clean, X_test_clean, y_train_clean = detect_and_handle_outliers(X_train, X_test)
    else:
        X_train_clean, X_test_clean = X_train.copy(), X_test.copy()
        y_train_clean = y_train
    
    # Step 2: Transform features
    X_train_trans, X_test_trans = transform_features(X_train_clean, X_test_clean)
    
    # Step 3: Create hardware-specific features
    X_train_hw, X_test_hw = engineer_hardware_features(X_train_trans, X_test_trans)
    
    # Step 4: Create polynomial features (only for numeric columns)
    numeric_cols = X_train_hw.select_dtypes(include=['number']).columns
    X_train_poly, X_test_poly = create_polynomial_features(
        X_train_hw[numeric_cols], 
        X_test_hw[numeric_cols]
    )
    
    # Step 5: Combine all features
    X_train_enhanced = pd.concat([X_train_hw, X_train_poly], axis=1)
    X_test_enhanced = pd.concat([X_test_hw, X_test_poly], axis=1)
    
    # Step 6: Feature selection (if y_train is available)
    if y_train_clean is not None:
        X_train_selected, X_test_selected, importance = select_features(
            X_train_enhanced, 
            X_test_enhanced, 
            y_train_clean,
            k=50  # Select top 50 features
        )
    else:
        X_train_selected, X_test_selected = X_train_enhanced, X_test_enhanced
        importance = None
    
    # Step 7: Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_selected),
        columns=X_train_selected.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_selected),
        columns=X_test_selected.columns
    )
    
    # Save scaler
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "standard_scaler.joblib"))
    
    # Create final enhanced dataset
    enhanced_data = {
        'X_train_original': X_train,
        'X_test_original': X_test,
        'y_train_original': y_train,
        'y_test_original': y_test,
        'X_train_enhanced': X_train_scaled,
        'X_test_enhanced': X_test_scaled,
        'y_train_enhanced': y_train_clean,
        'y_test_enhanced': y_test,
        'feature_importance': importance
    }
    
    # Save enhanced datasets
    X_train_scaled.to_csv(os.path.join(DATA_DIR, "X_train_enhanced.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(DATA_DIR, "X_test_enhanced.csv"), index=False)
    
    if y_train_clean is not None:
        pd.DataFrame(y_train_clean).to_csv(os.path.join(DATA_DIR, "y_train_enhanced.csv"), index=False)
    if y_test is not None:
        pd.DataFrame(y_test).to_csv(os.path.join(DATA_DIR, "y_test_enhanced.csv"), index=False)
    
    logger.info(f"Enhanced datasets created with {X_train_scaled.shape[1]} features")
    logger.info(f"Saved enhanced datasets to {DATA_DIR}")
    
    return enhanced_data

if __name__ == "__main__":
    # Load data
    data = load_data()
    
    # Create enhanced dataset
    enhanced_data = create_enhanced_dataset(
        data['X_train_numeric'],
        data['X_test_numeric'],
        data['y_train'],
        data['y_test']
    )
    
    logger.info("Feature engineering completed successfully!")
