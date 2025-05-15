#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering module for ML Marketplace project.

This module provides pipelines for preprocessing and feature engineering
for the computer dataset. It works with data that has already been cleaned
by data_cleaner.py, missing_data.py, and merge_features.py.
"""

import pandas as pd
import numpy as np
import joblib
import gc
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import os
import sys
from typing import List, Dict, Tuple, Optional, Union

# Add the current directory to path if not already there
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# We won't use the old preprocessing functions as we've created improved versions

class FeatureRatioTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for creating feature ratios.
    
    This creates additional features that capture relationships between
    numeric features, such as price per GB of RAM, etc.
    """
    
    def __init__(self, ratio_pairs=None):
        """
        Initialize with pairs of column names to create ratios.
        
        Args:
            ratio_pairs: List of tuples (col1, col2, name) where col1 and col2 are
                         column names and name is the name for the ratio feature.
        """
        if ratio_pairs is None:
            # Default ratio pairs - add more as needed
            self.ratio_pairs = [
                # Price per unit features
                ('Precio', 'RAM_Memoria RAM', 'price_per_ram_gb'),
                ('Precio', 'Medidas y peso_Peso', 'price_per_weight'),
                # Performance ratios
                ('Procesador_Frecuencia del procesador', 'RAM_Memoria RAM', 'processor_ram_ratio'),
                # Weight-to-performance ratio
                ('Medidas y peso_Peso', 'Procesador_Frecuencia del procesador', 'weight_to_processor_ratio'),
            ]
        else:
            self.ratio_pairs = ratio_pairs
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Only process if X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            return X
            
        X_copy = X.copy()
        
        # Create ratio features
        for num_col, denom_col, name in self.ratio_pairs:
            if num_col in X_copy.columns and denom_col in X_copy.columns:
                # Ensure denominator isn't zero
                mask = (X_copy[denom_col] != 0) & (~X_copy[denom_col].isna()) & (~X_copy[num_col].isna())
                
                if mask.sum() > 0:
                    X_copy.loc[mask, name] = X_copy.loc[mask, num_col] / X_copy.loc[mask, denom_col]
                    # Fill NaN values with median
                    if X_copy[name].isna().any():
                        X_copy[name] = X_copy[name].fillna(X_copy[name].median())
        
        return X_copy

class CategoricalBinningTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for binning categorical variables.
    
    This reduces cardinality of categorical features by grouping
    less frequent categories into an 'Other' category.
    """
    
    def __init__(self, min_frequency=0.05, columns=None):
        """
        Initialize with minimum frequency threshold.
        
        Args:
            min_frequency: Minimum frequency for a category to remain separate.
            columns: Columns to apply this transformer to.
        """
        self.min_frequency = min_frequency
        self.columns = columns
        self.value_maps = {}
    
    def fit(self, X, y=None):
        # Only process if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            if self.columns is None:
                self.columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            for col in self.columns:
                if col in X.columns:
                    # Calculate value counts
                    value_counts = X[col].value_counts(normalize=True)
                    
                    # Identify values that meet the threshold
                    keep_values = value_counts[value_counts >= self.min_frequency].index.tolist()
                    
                    # Create value map
                    self.value_maps[col] = {val: val if val in keep_values else 'Other' 
                                         for val in X[col].unique() if pd.notna(val)}
        
        return self
    
    def transform(self, X):
        # Only transform if X is a DataFrame and we have value_maps
        if isinstance(X, pd.DataFrame) and self.value_maps:
            X_copy = X.copy()
            
            for col, value_map in self.value_maps.items():
                if col in X_copy.columns:
                    X_copy[col] = X_copy[col].map(lambda x: value_map.get(x, 'Other') if pd.notna(x) else 'Unknown')
            
            return X_copy
        else:
            # If not a DataFrame, return X unchanged
            return X

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer for selecting the engineered feature set.
    """
    
    def __init__(self, features=None, drop_features=None):
        """
        Initialize with feature selection options.
        
        Args:
            features: List of features to include (if None, all non-dropped features are included)
            drop_features: List of features to exclude
        """
        self.features = features
        self.drop_features = drop_features if drop_features is not None else []
    
    def fit(self, X, y=None):
        if self.features is None:
            # Use all columns except those specified in drop_features
            if isinstance(X, pd.DataFrame):
                self.features = [col for col in X.columns if col not in self.drop_features]
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.features]
        else:
            # If X is numpy array, return as is (this is a simplification)
            return X

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer for extracting information from text fields.
    """
    
    def __init__(self, text_columns=None):
        self.text_columns = text_columns
    
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            # If not a DataFrame, just return self
            return self
            
        if self.text_columns is None:
            # Find columns that might contain textual descriptions
            try:
                self.text_columns = [col for col in X.columns 
                                   if col in X.select_dtypes(include=['object']).columns 
                                   and X[col].str.len().mean() > 15]
            except Exception as e:
                print(f"Error finding text columns: {e}")
                self.text_columns = []
                
        return self
    
    def transform(self, X):
        # Only process if X is a DataFrame and we have text columns
        if not isinstance(X, pd.DataFrame) or not self.text_columns:
            return X
            
        X_copy = X.copy()
        
        for col in self.text_columns:
            if col in X_copy.columns:
                # Extract some basic text features
                try:
                    # Word count
                    X_copy[f"{col}_word_count"] = X_copy[col].fillna('').astype(str).str.split().str.len()
                    
                    # Character count
                    X_copy[f"{col}_char_count"] = X_copy[col].fillna('').astype(str).str.len()
                    
                    # Uppercase letter count (may indicate emphasis)
                    X_copy[f"{col}_upper_count"] = X_copy[col].fillna('').astype(str).str.count(r'[A-Z]')
                except Exception as e:
                    print(f"Error processing {col}: {e}")
        
        return X_copy

def create_numeric_pipeline():
    """
    Create a pipeline for processing numeric features.
    
    Returns:
        Pipeline: A scikit-learn pipeline for numeric feature processing.
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Already handled missing values, but this is a safeguard
        ('scaler', RobustScaler()),  # Use RobustScaler to handle outliers better
        ('normalizer', PowerTransformer(method='yeo-johnson', standardize=True))  # Transform features closer to normal distribution
    ])

def create_categorical_pipeline(top_n_categories=10):
    """
    Create a pipeline for processing categorical features.
    
    Args:
        top_n_categories (int): Number of top categories to one-hot encode.
        
    Returns:
        Pipeline: A scikit-learn pipeline for categorical feature processing.
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),  # Already handled missing, but safeguard
        ('binning', CategoricalBinningTransformer(min_frequency=0.05)),  # Reduce cardinality
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore', max_categories=top_n_categories))
    ])

def get_numeric_features(df):
    """
    Get numeric feature column names from dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        list: List of numeric column names.
    """
    return df.select_dtypes(include=['number']).columns.tolist()

def get_categorical_features(df):
    """
    Get categorical feature column names from dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        list: List of categorical column names.
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

# Define selector functions for use in pipeline
def select_numeric_features(X, features):
    """Select numeric features from dataframe."""
    return X[features]

def select_categorical_features(X, features):
    """Select categorical features from dataframe."""
    return X[features]

def create_preprocessing_pipeline(df, include_advanced_features=True):
    """
    Create the complete preprocessing pipeline.
    
    Args:
        df (pd.DataFrame): Input dataframe for identifying column types.
        include_advanced_features (bool): Whether to include advanced feature engineering.
        
    Returns:
        Pipeline: A scikit-learn pipeline for complete preprocessing.
    """
    # Create a simpler pipeline structure to avoid issues with data type conversions
    
    # Get numeric and categorical feature lists
    numeric_features = get_numeric_features(df)
    categorical_features = get_categorical_features(df)
    
    # Ensure we're working with lists, not pandas Index
    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)
    
    # Basic pipeline without complex transformers
    # This will be more stable but have fewer engineered features
    steps = [
        # Only do the basic numeric and categorical processing
        ('features', FeatureUnion([
            ('num', Pipeline([
                ('selector', FunctionTransformer(lambda X: X[numeric_features] if isinstance(X, pd.DataFrame) else X[:, :len(numeric_features)])),
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])),
            ('cat', Pipeline([
                ('selector', FunctionTransformer(lambda X: X[categorical_features] if isinstance(X, pd.DataFrame) else X[:, len(numeric_features):])),
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ]))
        ])),
    ]
    
    # Only add PCA if we have enough features to make it worthwhile
    if len(numeric_features) + len(categorical_features) > 30:
        steps.append(('dim_reduction', PCA(n_components=0.95)))
    
    # Add feature selection if target is present
    try:
        if 'Precio' in df.columns:
            steps.append(('feature_selection', SelectKBest(f_regression, k=min(20, len(numeric_features) + len(categorical_features)))))
    except Exception as e:
        print(f"Skipping feature selection: {e}")
    
    # Create the pipeline
    preprocessing_pipeline = Pipeline(steps)
    
    return preprocessing_pipeline

def load_processed_data(file_path=None):
    """
    Load the processed data file.
    
    Args:
        file_path (str): Path to the processed data file.
        
    Returns:
        pd.DataFrame: The processed dataframe.
    """
    if file_path is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        file_path = os.path.join(data_dir, "db_computers_merged.csv")
    
    return pd.read_csv(file_path, encoding='utf-8-sig')

def process_and_save_features(chunk_size=1000, target_col='Precio'):
    """
    Process the already cleaned data and save engineered features.
    
    This function loads data in chunks, applies feature engineering, and saves results.
    
    Args:
        chunk_size (int): Size of chunks to process.
        target_col (str): Target column for regression (used for feature selection).
        
    Returns:
        None: Results are saved to disk.
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    input_file = os.path.join(data_dir, "db_computers_merged.csv")
    output_file = os.path.join(data_dir, "db_computers_features.csv")
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    pipeline_path = os.path.join(model_dir, "preprocessing_pipeline.joblib")
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Loading sample data to build pipeline...")
    df_sample = pd.read_csv(input_file, nrows=5000, encoding='utf-8-sig')
    
    # Separate target if available
    y_sample = None
    if target_col in df_sample.columns:
        y_sample = df_sample[target_col]
    
    # Create and fit the pipeline
    print("Creating feature engineering pipeline...")
    pipeline = create_preprocessing_pipeline(df_sample)
    print("Fitting pipeline on sample data...")
    pipeline.fit(df_sample, y_sample)
    
    # Save the fitted pipeline
    print(f"Saving pipeline to {pipeline_path}")
    joblib.dump(pipeline, pipeline_path)
    
    # Process data in chunks
    print("Processing full dataset in chunks...")
    first_chunk = True
    
    for chunk_num, chunk in enumerate(pd.read_csv(input_file, 
                                               chunksize=chunk_size, 
                                               encoding='utf-8-sig')):
        print(f"Processing chunk {chunk_num+1}...")
        
        # Transform chunk
        transformed_chunk = pipeline.transform(chunk)
        
        # Convert to DataFrame with proper column names
        column_names = [f"feature_{i}" for i in range(transformed_chunk.shape[1])]
        transformed_df = pd.DataFrame(transformed_chunk, columns=column_names)
        
        # Add target column if available
        if target_col in chunk.columns:
            transformed_df[target_col] = chunk[target_col].values
        
        # Write to output file
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        transformed_df.to_csv(output_file, index=False, mode=mode, header=header)
        
        if first_chunk:
            first_chunk = False
        
        # Free memory
        del chunk
        del transformed_chunk
        del transformed_df
        gc.collect()
    
    print(f"Feature engineering complete. Engineered features saved to {output_file}")
    print(f"Preprocessing pipeline saved to {pipeline_path}")

def transform_with_pipeline(X, pipeline_path=None):
    """
    Transform data using the saved preprocessing pipeline.
    
    Args:
        X (pd.DataFrame): Input dataframe.
        pipeline_path (str): Path to the saved pipeline. If None, will try to find it in standard location.
        
    Returns:
        pd.DataFrame: Transformed dataframe with engineered features.
    """
    # Determine proper path for loading
    if pipeline_path is None:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        pipeline_path = os.path.join(model_dir, "preprocessing_pipeline.joblib")
    
    # Load the pipeline
    try:
        pipeline = joblib.load(pipeline_path)
        print(f"Loaded preprocessing pipeline from {pipeline_path}")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("Creating new pipeline...")
        pipeline = create_preprocessing_pipeline(X)
        pipeline.fit(X)
    
    # Transform the data
    X_transformed = pipeline.transform(X)
    
    # Convert to DataFrame with feature names
    if isinstance(X_transformed, np.ndarray):
        X_transformed = pd.DataFrame(
            X_transformed, 
            columns=[f'feature_{i}' for i in range(X_transformed.shape[1])]
        )
    
    return X_transformed

def fit_transform(X_raw):
    """
    Fit the preprocessing pipeline to the raw data and transform it.
    
    Args:
        X_raw (pd.DataFrame): Raw input dataframe.
        
    Returns:
        pd.DataFrame: Transformed dataframe with engineered features.
    """
    pipeline = create_preprocessing_pipeline(X_raw)
    X_transformed = pipeline.fit_transform(X_raw)
    
    # Save fitted pipeline
    # Determine proper path for saving
    if os.path.exists('../models'):
        save_path = '../models/preprocessing_pipeline.joblib'
    else:
        save_path = 'models/preprocessing_pipeline.joblib'
    
    # Make sure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    joblib.dump(pipeline, save_path)
    
    return X_transformed

def transform(X_raw):
    """
    Transform raw data using the saved preprocessing pipeline.
    
    Args:
        X_raw (pd.DataFrame): Raw input dataframe.
        
    Returns:
        pd.DataFrame: Transformed dataframe with engineered features.
    """
    try:
        # Try to find the pipeline file
        if os.path.exists('../models/preprocessing_pipeline.joblib'):
            pipeline_path = '../models/preprocessing_pipeline.joblib'
        else:
            pipeline_path = 'models/preprocessing_pipeline.joblib'
            
        pipeline = joblib.load(pipeline_path)
        return pipeline.transform(X_raw)
    except FileNotFoundError:
        print("Pipeline not found. Run fit_transform first.")
        return None

if __name__ == "__main__":
    # Process the data and save engineered features
    print("Starting feature engineering process...")
    process_and_save_features(chunk_size=1000, target_col='Precio')
    
    print("Done!")
