#!/usr/bin/env python3
"""
Memory-Efficient Model Preparation Module for ML Marketplace.

This module performs final preparation of the engineered features before modeling
with a focus on minimizing memory usage through chunk processing.
"""

import os
import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
import joblib
from typing import List, Dict, Optional, Union, Tuple

# Set up paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "model_prep")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def find_low_variance_features(input_file, chunk_size=1000, threshold=0.01):
    """
    Find features with variance below the threshold.
    
    Args:
        input_file: Path to input CSV
        chunk_size: Size of chunks for processing
        threshold: Variance threshold
        
    Returns:
        List of low variance feature names
    """
    print("Finding low variance features...")
    
    # Initialize variances as empty Series
    variances = pd.Series(dtype='float64')
    
    # Process chunks
    chunk_count = 0
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        chunk_count += 1
        print(f"  Processing chunk {chunk_count}...")
        
        # Calculate variances for this chunk
        chunk_variances = chunk.var()
        
        # Update running variances
        if variances.empty:
            variances = chunk_variances
        else:
            # Combine variances (approximate method)
            variances = (variances + chunk_variances) / 2
        
        # Clean up
        del chunk
        gc.collect()
    
    # Filter low variance features
    low_var_features = variances[variances < threshold].index.tolist()
    
    print(f"Found {len(low_var_features)} low variance features")
    
    # Save list of low variance features
    pd.Series(low_var_features, name='feature').to_csv(
        os.path.join(OUTPUT_DIR, "low_variance_features.csv"), 
        index=False
    )
    
    return low_var_features

def create_train_test_split_indices(input_file, test_size=0.2, random_state=42):
    """
    Create indices for train/test split without loading entire dataset.
    
    Args:
        input_file: Path to input CSV
        test_size: Proportion of data for test set
        random_state: Random seed
        
    Returns:
        Dictionary with train and test indices
    """
    print("Creating train/test split indices...")
    
    # Count total rows
    row_count = 0
    for chunk in pd.read_csv(input_file, chunksize=1000, usecols=[0]):
        row_count += len(chunk)
    
    # Create indices
    all_indices = np.arange(row_count)
    train_indices, test_indices = train_test_split(
        all_indices, test_size=test_size, random_state=random_state
    )
    
    # Save indices
    np.save(os.path.join(OUTPUT_DIR, "train_indices.npy"), train_indices)
    np.save(os.path.join(OUTPUT_DIR, "test_indices.npy"), test_indices)
    
    print(f"Created train set ({len(train_indices)} samples) and test set ({len(test_indices)} samples)")
    
    return {
        'train': train_indices,
        'test': test_indices
    }

def create_train_test_files(input_file, output_dir, indices, target_col='Precio', 
                           drop_features=None, chunk_size=1000):
    """
    Create train and test files by processing data in chunks.
    
    Args:
        input_file: Path to input CSV
        output_dir: Directory for output files
        indices: Dictionary with train and test indices
        target_col: Target column name
        drop_features: Features to drop
        chunk_size: Size of chunks for processing
    """
    print("Creating train and test files...")
    
    # Set up output files
    train_file = os.path.join(output_dir, "train_set.csv")
    test_file = os.path.join(output_dir, "test_set.csv")
    
    # Convert indices to sets for faster lookup
    train_indices_set = set(indices['train'])
    test_indices_set = set(indices['test'])
    
    # Initialize variables
    first_train_chunk = True
    first_test_chunk = True
    
    # Process chunks
    for chunk_idx, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        print(f"  Processing chunk {chunk_idx + 1}...")
        
        # Calculate row indices for this chunk
        start_idx = chunk_idx * chunk_size
        chunk_indices = range(start_idx, start_idx + len(chunk))
        
        # Prepare train and test dataframes
        train_chunk = pd.DataFrame()
        test_chunk = pd.DataFrame()
        
        # Split rows into train and test
        for i, row_idx in enumerate(chunk_indices):
            if row_idx in train_indices_set:
                train_chunk = pd.concat([train_chunk, chunk.iloc[[i]]])
            elif row_idx in test_indices_set:
                test_chunk = pd.concat([test_chunk, chunk.iloc[[i]]])
        
        # Drop features if specified
        if drop_features:
            drop_cols = [col for col in drop_features if col in train_chunk.columns]
            if drop_cols:
                train_chunk = train_chunk.drop(columns=drop_cols)
                test_chunk = test_chunk.drop(columns=drop_cols)
        
        # Save train chunk
        if not train_chunk.empty:
            train_chunk.to_csv(
                train_file, 
                mode='w' if first_train_chunk else 'a',
                header=first_train_chunk,
                index=False
            )
            if first_train_chunk:
                first_train_chunk = False
        
        # Save test chunk
        if not test_chunk.empty:
            test_chunk.to_csv(
                test_file, 
                mode='w' if first_test_chunk else 'a',
                header=first_test_chunk,
                index=False
            )
            if first_test_chunk:
                first_test_chunk = False
        
        # Clean up
        del chunk, train_chunk, test_chunk
        gc.collect()
    
    print(f"Train and test files created at:")
    print(f"  {train_file}")
    print(f"  {test_file}")

def prepare_for_modeling(input_file=None, target_col='Precio'):
    """
    Main function to prepare data for modeling.
    
    Args:
        input_file: Path to engineered features file
        target_col: Name of target column
    """
    print("Starting memory-efficient model preparation...")
    
    # Set input file if not provided
    if input_file is None:
        input_file = os.path.join(DATA_DIR, "db_computers_features.csv")
    
    # Find low variance features
    low_var_features = find_low_variance_features(input_file)
    
    # Create train/test split indices
    indices = create_train_test_split_indices(input_file)
    
    # Create train and test files
    create_train_test_files(
        input_file=input_file,
        output_dir=DATA_DIR,
        indices=indices,
        target_col=target_col,
        drop_features=low_var_features
    )
    
    print("\nModel preparation complete!")
    print(f"You can now train models using the train_set.csv file and evaluate on test_set.csv")

if __name__ == "__main__":
    # Run the memory-efficient preparation pipeline
    prepare_for_modeling()
    
    # Final message
    print("\nData is ready for modeling!")
    print(f"Train and test sets saved to {DATA_DIR}")
