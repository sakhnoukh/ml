"""
Feature engineering module for the ML Marketplace.

This module handles feature engineering, selection, and transformation of the preprocessed
computer dataset. It creates a feature pipeline that can be reused for training and inference.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Custom transformer for encoding multi-label categorical features.
    
    Takes the first label as primary, one-hot encodes top categories, 
    and groups rare categories as "Other".
    """
    
    def __init__(self, top_n: int = 10, delimiter: str = ',', other_label: str = 'Other'):
        """
        Initialize the transformer.
        
        Args:
            top_n: Number of top categories to one-hot encode
            delimiter: Delimiter used to separate multiple labels
            other_label: Label to use for rare categories
        """
        self.top_n = top_n
        self.delimiter = delimiter
        self.other_label = other_label
        self.top_categories_ = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer by identifying top categories for each column.
        
        Args:
            X: DataFrame with categorical columns
            y: Ignored
            
        Returns:
            self
        """
        X_copy = X.copy()
        
        for col in X_copy.columns:
            # Extract first label from each multi-label value
            X_copy[col] = X_copy[col].astype(str).apply(
                lambda x: x.split(self.delimiter)[0].strip() if pd.notna(x) else np.nan
            )
            
            # Count frequencies and identify top categories
            value_counts = X_copy[col].value_counts()
            self.top_categories_[col] = list(value_counts.head(self.top_n).index)
            
        return self
    
    def transform(self, X: pd.DataFrame):
        """
        Transform multi-label categorical columns.
        
        Args:
            X: DataFrame with categorical columns
            
        Returns:
            Transformed DataFrame
        """
        X_transformed = X.copy()
        
        for col in X_transformed.columns:
            if col in self.top_categories_:
                # Extract first label from each multi-label value
                X_transformed[col] = X_transformed[col].astype(str).apply(
                    lambda x: x.split(self.delimiter)[0].strip() if pd.notna(x) else np.nan
                )
                
                # Replace rare categories with "Other"
                X_transformed[col] = X_transformed[col].apply(
                    lambda x: x if x in self.top_categories_[col] else self.other_label
                )
        
        return X_transformed


def identify_core_features(df: pd.DataFrame, 
                          numeric_features: List[str] = None,
                          categorical_features: List[str] = None) -> Tuple[List[str], List[str]]:
    """
    Identify core features for the model.
    
    Args:
        df: DataFrame containing the processed data
        numeric_features: List of numeric features to include
        categorical_features: List of categorical features to include
        
    Returns:
        Tuple of (numeric_features, categorical_features)
    """
    # Default core features if none provided
    if numeric_features is None:
        # Typical computer specs numeric features
        numeric_features = [
            col for col in df.columns if df[col].dtype.kind in 'ifc' and 
            any(keyword in col.lower() for keyword in 
                ['processor', 'speed', 'ram', 'storage', 'screen', 'size', 'weight', 
                 'resolution', 'battery', 'price', 'year', 'cores'])
        ]
    
    if categorical_features is None:
        # Typical computer specs categorical features
        categorical_features = [
            col for col in df.columns if df[col].dtype.kind not in 'ifc' and
            any(keyword in col.lower() for keyword in 
                ['brand', 'os', 'gpu', 'type', 'resolution', 'model', 'category', 
                 'processor', 'cpu'])
        ]
    
    logger.info(f"Identified {len(numeric_features)} numeric features: {numeric_features}")
    logger.info(f"Identified {len(categorical_features)} categorical features: {categorical_features}")
    
    return numeric_features, categorical_features


def create_feature_pipeline(numeric_features: List[str], 
                           categorical_features: List[str], 
                           top_n_categories: int = 10) -> Pipeline:
    """
    Create a feature engineering pipeline.
    
    Args:
        numeric_features: List of numeric features
        categorical_features: List of categorical features
        top_n_categories: Number of top categories to one-hot encode
        
    Returns:
        sklearn Pipeline for feature transformation
    """
    logger.info("Creating feature engineering pipeline")
    
    # Numeric pipeline with scaling
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline with multi-label encoding and one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('multi_label', MultiLabelEncoder(top_n=top_n_categories)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop other columns
    )
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    
    return pipeline


def get_feature_names(pipeline: Pipeline, numeric_features: List[str], 
                     categorical_features: List[str]) -> List[str]:
    """
    Get feature names after transformation.
    
    Args:
        pipeline: Trained feature pipeline
        numeric_features: List of numeric features
        categorical_features: List of categorical features
        
    Returns:
        List of transformed feature names
    """
    # Get feature names from one-hot encoding
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Get the onehotencoder for categorical features
    onehotencoder = preprocessor.transformers_[1][1].named_steps['onehot']
    
    # Get categories for each categorical feature
    cat_feature_names = []
    for i, feature in enumerate(categorical_features):
        cat_names = [f"{feature}_{cat}" for cat in onehotencoder.categories_[i]]
        cat_feature_names.extend(cat_names)
    
    # Combine with numeric feature names
    feature_names = numeric_features + cat_feature_names
    
    return feature_names


def save_feature_pipeline(pipeline: Pipeline, output_path: str):
    """
    Save feature engineering pipeline.
    
    Args:
        pipeline: Trained feature pipeline
        output_path: Path to save the pipeline
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Saving feature pipeline to {output_path}")
    joblib.dump(pipeline, output_path)


def load_feature_pipeline(input_path: str) -> Pipeline:
    """
    Load feature engineering pipeline.
    
    Args:
        input_path: Path to the saved pipeline
        
    Returns:
        Loaded pipeline
    """
    logger.info(f"Loading feature pipeline from {input_path}")
    return joblib.load(input_path)


def engineer_features(df: pd.DataFrame, numeric_features: List[str] = None, 
                     categorical_features: List[str] = None, 
                     pipeline_path: Optional[str] = None) -> Tuple[np.ndarray, Pipeline]:
    """
    Full feature engineering process.
    
    Args:
        df: DataFrame containing the processed data
        numeric_features: List of numeric features to include
        categorical_features: List of categorical features to include
        pipeline_path: Path to save the feature pipeline
        
    Returns:
        Tuple of (transformed features, feature pipeline)
    """
    # Identify core features if not provided
    if numeric_features is None or categorical_features is None:
        numeric_features, categorical_features = identify_core_features(df)
    
    # Create feature pipeline
    feature_pipeline = create_feature_pipeline(numeric_features, categorical_features)
    
    # Transform data
    logger.info("Transforming features")
    X_transformed = feature_pipeline.fit_transform(df)
    
    # Save pipeline if path provided
    if pipeline_path:
        save_feature_pipeline(feature_pipeline, pipeline_path)
    
    return X_transformed, feature_pipeline


if __name__ == "__main__":
    # Example usage
    # from preprocessing import preprocess_data
    # 
    # # Preprocess data
    # df = preprocess_data("../data/db_computers_2025_raw.csv")
    # 
    # # Engineer features
    # X_transformed, pipeline = engineer_features(
    #     df, 
    #     pipeline_path="../models/feature_pipeline.joblib"
    # )
    pass
