#!/usr/bin/env python3
"""
Feature Builder for ML Marketplace.

This module provides a simpler approach to feature engineering that
avoids complex pipelines and directly transforms data.
"""

import os
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import joblib
from typing import List, Dict, Optional, Union, Tuple

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

def get_numeric_features(df: pd.DataFrame) -> List[str]:
    """
    Get numeric feature column names from dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        list: List of numeric column names.
    """
    return df.select_dtypes(include=['number']).columns.tolist()

def get_categorical_features(df: pd.DataFrame) -> List[str]:
    """
    Get categorical feature column names from dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        list: List of categorical column names.
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def encode_categorical_features(df: pd.DataFrame, categorical_cols: List[str], 
                               min_frequency: float = 0.05) -> pd.DataFrame:
    """
    One-hot encode categorical features with binning for rare categories.
    
    Args:
        df: Input dataframe
        categorical_cols: List of categorical column names
        min_frequency: Minimum frequency for a category to remain separate
        
    Returns:
        Encoded dataframe with one-hot encoded features
    """
    # Output dataframe
    encoded_df = pd.DataFrame(index=df.index)
    
    # Process each categorical column
    for col in categorical_cols:
        if col not in df.columns:
            continue
            
        # Calculate value counts
        value_counts = df[col].value_counts(normalize=True)
        
        # Identify values that meet the threshold
        keep_values = value_counts[value_counts >= min_frequency].index.tolist()
        
        # Replace rare categories with 'Other'
        temp_col = df[col].copy()
        temp_col = temp_col.apply(lambda x: x if x in keep_values else 'Other')
        
        # One-hot encode
        dummies = pd.get_dummies(temp_col, prefix=col, dummy_na=True)
        
        # Add to encoded dataframe
        encoded_df = pd.concat([encoded_df, dummies], axis=1)
    
    return encoded_df

def process_features_in_chunks(input_file: str, output_file: str, 
                              chunk_size: int = 1000, 
                              target_col: str = 'Precio') -> None:
    """
    Process features in chunks without using scikit-learn pipelines.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        chunk_size: Size of chunks to process
        target_col: Name of target column
        use_pca: Whether to apply PCA for dimensionality reduction
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read a sample to identify column types
    print(f"Loading sample data to identify features...")
    df_sample = pd.read_csv(input_file, nrows=5000, encoding='utf-8-sig')
    
    # Get column types
    numeric_cols = get_numeric_features(df_sample)
    categorical_cols = get_categorical_features(df_sample)
    
    print(f"Found {len(numeric_cols)} numeric features and {len(categorical_cols)} categorical features")
    
    # Create scalers for numeric features
    print("Fitting scalers on sample data...")
    scaler = RobustScaler()
    scaler.fit(df_sample[numeric_cols])
    
    # Save scaler
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(model_dir, exist_ok=True)
    scaler_path = os.path.join(model_dir, "robust_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    # Pre-compute categorical encoding on sample
    print("Encoding categorical features...")
    encoded_categorical = encode_categorical_features(df_sample, categorical_cols)
    encoded_feature_names = encoded_categorical.columns.tolist()
    
    # NOTE: We're skipping PCA because of feature name inconsistency issues between chunks
    print("Skipping PCA to avoid feature name inconsistency issues between chunks.")
    
    # Clean up memory
    del df_sample
    del encoded_categorical
    gc.collect()
    
    # Process data in chunks
    print("Processing data in chunks...")
    first_chunk = True
    
    # For each chunk
    for chunk_num, chunk in enumerate(pd.read_csv(input_file, 
                                               chunksize=chunk_size, 
                                               encoding='utf-8-sig')):
        print(f"Processing chunk {chunk_num+1}...")
        
        # Extract target if available
        target = None
        if target_col in chunk.columns:
            target = chunk[target_col].copy()
        
        # Process numeric features
        if numeric_cols:
            # Scale numeric features
            scaled_numeric = pd.DataFrame(
                scaler.transform(chunk[numeric_cols]), 
                columns=numeric_cols,
                index=chunk.index
            )
        else:
            scaled_numeric = pd.DataFrame(index=chunk.index)
        
        # Process categorical features
        encoded_categorical = encode_categorical_features(chunk, categorical_cols)
        
        # Combine features
        processed_features = pd.concat([scaled_numeric, encoded_categorical], axis=1)
        
        # Fill NaN values
        processed_features = processed_features.fillna(0)
        
        # Add target column back if available
        if target is not None:
            processed_features[target_col] = target
        
        # Write to output file
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        processed_features.to_csv(output_file, index=False, mode=mode, header=header)
        
        if first_chunk:
            first_chunk = False
        
        # Clean up memory
        del chunk
        del scaled_numeric
        del encoded_categorical
        del processed_features
        if target is not None:
            del target
        gc.collect()
    
    print(f"Feature engineering complete. Engineered features saved to {output_file}")

if __name__ == "__main__":
    # Process and save features
    print("Starting feature engineering process...")
    
    # Input and output paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    input_file = os.path.join(data_dir, "db_computers_merged.csv")
    output_file = os.path.join(data_dir, "db_computers_features.csv")
    
    # Process features
    process_features_in_chunks(
        input_file=input_file,
        output_file=output_file,
        chunk_size=1000,
        target_col='Precio'
    )
    
    print("Done!")
