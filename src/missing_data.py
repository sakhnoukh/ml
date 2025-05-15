#!/usr/bin/env python3
"""
Missing Data Handling for ML Marketplace.

This module provides efficient functions to handle missing data in the computer dataset
without causing memory explosion.
"""

import os
import pandas as pd
import numpy as np
import gc
from typing import List, Dict, Tuple, Optional, Union

def analyze_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing data patterns in the dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with missing data statistics
    """
    # Calculate missing percentage for each column
    missing_stats = pd.DataFrame({
        'missing_count': df.isna().sum(),
        'missing_percent': df.isna().mean() * 100,
        'dtype': df.dtypes
    })
    
    # Sort by missing percentage (descending)
    missing_stats = missing_stats.sort_values('missing_percent', ascending=False)
    
    return missing_stats

def identify_columns_for_treatment(missing_stats: pd.DataFrame, 
                                  high_threshold: float = 70.0,
                                  medium_threshold: float = 30.0) -> Dict[str, List[str]]:
    """
    Identify columns for different missing data treatments based on missing percentage.
    
    Args:
        missing_stats: DataFrame with missing data statistics
        high_threshold: Percentage threshold for high missing values (to drop)
        medium_threshold: Percentage threshold for medium missing values (to flag)
        
    Returns:
        Dictionary with column groups for different treatments
    """
    # Group columns based on missing percentage
    high_missing = missing_stats[missing_stats['missing_percent'] > high_threshold].index.tolist()
    medium_missing = missing_stats[(missing_stats['missing_percent'] <= high_threshold) & 
                                 (missing_stats['missing_percent'] > medium_threshold)].index.tolist()
    low_missing = missing_stats[missing_stats['missing_percent'] <= medium_threshold].index.tolist()
    
    # Separate numeric and categorical columns
    numeric_medium = [col for col in medium_missing 
                     if pd.api.types.is_numeric_dtype(missing_stats.loc[col, 'dtype'])]
    categorical_medium = [col for col in medium_missing 
                         if not pd.api.types.is_numeric_dtype(missing_stats.loc[col, 'dtype'])]
    
    numeric_low = [col for col in low_missing 
                  if pd.api.types.is_numeric_dtype(missing_stats.loc[col, 'dtype'])]
    categorical_low = [col for col in low_missing 
                      if not pd.api.types.is_numeric_dtype(missing_stats.loc[col, 'dtype'])]
    
    return {
        'drop': high_missing,
        'flag_numeric': numeric_medium,
        'flag_categorical': categorical_medium,
        'impute_numeric': numeric_low,
        'impute_categorical': categorical_low
    }

def handle_missing_data_in_chunks(input_file: str, 
                                 output_file: str,
                                 column_groups: Dict[str, List[str]],
                                 numeric_impute_values: Optional[Dict[str, float]] = None,
                                 categorical_impute_values: Optional[Dict[str, str]] = None,
                                 chunk_size: int = 1000) -> None:
    """
    Handle missing data in chunks to avoid memory explosion.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        column_groups: Dictionary with column groups for different treatments
        numeric_impute_values: Dictionary with imputation values for numeric columns
        categorical_impute_values: Dictionary with imputation values for categorical columns
        chunk_size: Size of chunks to process
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Initialize variables
    first_chunk = True
    columns_to_keep = (column_groups['flag_numeric'] + column_groups['flag_categorical'] +
                      column_groups['impute_numeric'] + column_groups['impute_categorical'])
    
    # Process in chunks
    for chunk_num, chunk in enumerate(pd.read_csv(input_file, 
                                                chunksize=chunk_size, 
                                                usecols=columns_to_keep,
                                                encoding='utf-8-sig')):
        print(f"Processing chunk {chunk_num+1}...")
        
        # Create missing flags for columns with medium missing rate
        for col in column_groups['flag_numeric'] + column_groups['flag_categorical']:
            flag_col = f"{col}_missing"
            chunk[flag_col] = chunk[col].isna().astype(int)
        
        # Impute numeric columns with low missing rate
        for col in column_groups['impute_numeric']:
            if col in chunk.columns:
                # Get imputation value
                if numeric_impute_values and col in numeric_impute_values:
                    value = numeric_impute_values[col]
                else:
                    # Default to median
                    value = chunk[col].median()
                    if pd.isna(value):  # If median is NaN, use 0
                        value = 0.0
                
                # Impute missing values
                chunk[col] = chunk[col].fillna(value)
        
        # Impute categorical columns with low missing rate
        for col in column_groups['impute_categorical']:
            if col in chunk.columns:
                # Get imputation value
                if categorical_impute_values and col in categorical_impute_values:
                    value = categorical_impute_values[col]
                else:
                    # Default to "Unknown"
                    value = "Unknown"
                
                # Impute missing values
                chunk[col] = chunk[col].fillna(value)
        
        # Write to output file
        if first_chunk:
            chunk.to_csv(output_file, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(output_file, index=False, mode='a', header=False)
        
        # Free memory
        del chunk
        gc.collect()
    
    print(f"Missing data handling complete. Processed data saved to {output_file}")

def calculate_imputation_values(input_file: str, 
                               column_groups: Dict[str, List[str]]) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Calculate imputation values for numeric and categorical columns.
    
    Args:
        input_file: Path to input CSV file
        column_groups: Dictionary with column groups for different treatments
        
    Returns:
        Tuple of (numeric_impute_values, categorical_impute_values)
    """
    # Read a small sample to analyze
    impute_columns = column_groups['impute_numeric'] + column_groups['impute_categorical']
    df_sample = pd.read_csv(input_file, usecols=impute_columns, nrows=5000, encoding='utf-8-sig')
    
    # Calculate imputation values for numeric columns
    numeric_impute_values = {}
    for col in column_groups['impute_numeric']:
        if col in df_sample.columns:
            # Try to convert to numeric if not already
            if not pd.api.types.is_numeric_dtype(df_sample[col]):
                df_sample[col] = pd.to_numeric(df_sample[col], errors='coerce')
            
            # Calculate median
            median_val = df_sample[col].median()
            if not pd.isna(median_val):
                numeric_impute_values[col] = median_val
            else:
                numeric_impute_values[col] = 0.0
    
    # Calculate imputation values for categorical columns
    categorical_impute_values = {}
    for col in column_groups['impute_categorical']:
        if col in df_sample.columns:
            # Get most frequent value
            most_frequent = df_sample[col].mode().iloc[0] if not df_sample[col].mode().empty else "Unknown"
            categorical_impute_values[col] = most_frequent
    
    # Free memory
    del df_sample
    gc.collect()
    
    return numeric_impute_values, categorical_impute_values

def process_dataset(input_file: str, output_file: str, missing_threshold: float = 70.0) -> None:
    """
    Process the entire dataset with missing data handling.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        missing_threshold: Percentage threshold for high missing values (to drop)
    """
    print(f"Analyzing missing data patterns in {input_file}...")
    
    # Analyze a sample to determine missing patterns
    sample_df = pd.read_csv(input_file, nrows=5000, encoding='utf-8-sig')
    missing_stats = analyze_missing_data(sample_df)
    
    # Print missing data summary
    print("\nMissing Data Summary:")
    print(f"Total columns: {len(missing_stats)}")
    print(f"Columns with >70% missing: {len(missing_stats[missing_stats['missing_percent'] > 70])}")
    print(f"Columns with 30-70% missing: {len(missing_stats[(missing_stats['missing_percent'] > 30) & (missing_stats['missing_percent'] <= 70)])}")
    print(f"Columns with <30% missing: {len(missing_stats[missing_stats['missing_percent'] <= 30])}")
    
    # Identify columns for different treatments
    column_groups = identify_columns_for_treatment(missing_stats, 
                                                 high_threshold=missing_threshold,
                                                 medium_threshold=30.0)
    
    print("\nColumn Treatment Groups:")
    for group, cols in column_groups.items():
        print(f"{group}: {len(cols)} columns")
    
    # Calculate imputation values
    print("\nCalculating imputation values...")
    numeric_impute_values, categorical_impute_values = calculate_imputation_values(
        input_file, column_groups)
    
    # Handle missing data in chunks
    print("\nHandling missing data in chunks...")
    handle_missing_data_in_chunks(
        input_file=input_file,
        output_file=output_file,
        column_groups=column_groups,
        numeric_impute_values=numeric_impute_values,
        categorical_impute_values=categorical_impute_values,
        chunk_size=1000
    )

if __name__ == "__main__":
    # Example usage
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    # Use the cleaned data file (with unified units) as input
    input_file = os.path.join(data_dir, "db_computers_cleaned_units.csv")
    output_file = os.path.join(data_dir, "db_computers_final.csv")
    
    # Process the dataset
    process_dataset(input_file, output_file, missing_threshold=70.0)
