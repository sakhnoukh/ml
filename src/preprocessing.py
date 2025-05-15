#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocessing module for ML Marketplace project.

This module handles data loading and cleaning for the computer dataset.
"""

import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(filepath=None):
    """
    Load raw computer data from CSV file.
    
    Args:
        filepath (str, optional): Path to the CSV file. If None, defaults to 'data/db_computers_2025_raw.csv'
                                 relative to the project root.
        
    Returns:
        pd.DataFrame: Raw dataframe with computer data.
    """
    if filepath is None:
        # Determine if running from src directory or project root
        import os
        if os.path.exists('../data/db_computers_2025_raw.csv'):
            filepath = '../data/db_computers_2025_raw.csv'
        else:
            filepath = 'data/db_computers_2025_raw.csv'
    
    return pd.read_csv(filepath, encoding='utf-8-sig', low_memory=False)

def strip_text_from_numeric(df, columns=None):
    """
    Strip text from numeric columns and convert to float.
    Unify units (e.g., GB→GB, TB→1024 GB).
    
    Args:
        df (pd.DataFrame): Input dataframe.
        columns (list, optional): List of columns to process. If None, all numeric-like columns are processed.
        
    Returns:
        pd.DataFrame: Dataframe with cleaned numeric columns.
    """
    df_copy = df.copy()
    
    # If no columns specified, try to detect numeric-like columns
    if columns is None:
        # Look for columns that contain units or numeric patterns
        pattern = r'.*(\d+\.?\d*|\d*\.\d+)\s*(GB|TB|MB|MHz|GHz|inches|in|")'
        columns = [col for col in df.columns if df[col].astype(str).str.contains(pattern, regex=True, na=False).any()]
    
    for col in columns:
        if col in df.columns:
            # Skip if column is already numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            # Create a new column for processing
            df_copy[col] = df[col].astype(str)
            
            # Extract numeric values and convert TB to GB (1 TB = 1024 GB)
            def convert_value(val):
                if pd.isna(val) or val == 'nan':
                    return np.nan
                    
                val = str(val).lower()
                
                # Extract numeric part
                num_match = re.search(r'(\d+\.?\d*|\d*\.\d+)', val)
                if not num_match:
                    return np.nan
                    
                num_val = float(num_match.group(1))
                
                # Convert based on units
                if 'tb' in val:
                    return num_val * 1024  # Convert TB to GB
                elif 'gb' in val:
                    return num_val
                elif 'mb' in val:
                    return num_val / 1024  # Convert MB to GB
                elif 'ghz' in val:
                    return num_val
                elif 'mhz' in val:
                    return num_val / 1000  # Convert MHz to GHz
                elif any(unit in val for unit in ['inches', 'in', '"']):
                    return num_val  # Keep as is, these are screen sizes
                else:
                    return num_val
            
            df_copy[col] = df_copy[col].apply(convert_value)
    
    return df_copy

def parse_multi_label_fields(df, columns=None, delimiter=';'):
    """
    Parse multi-label text fields (e.g., "Windows 10; Office 365").
    Split by delimiter, extract primary label, and store full list.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        columns (list, optional): List of columns to process. If None, columns with ';' are processed.
        delimiter (str): Delimiter used in multi-label fields.
        
    Returns:
        pd.DataFrame: Dataframe with parsed multi-label columns.
    """
    df_copy = df.copy()
    
    # If no columns specified, detect columns with the delimiter
    if columns is None:
        columns = [col for col in df.columns if 
                   df[col].astype(str).str.contains(delimiter, regex=False, na=False).any()]
    
    for col in columns:
        if col in df.columns:
            # Skip non-string columns
            if not pd.api.types.is_string_dtype(df[col]):
                df_copy[col] = df_copy[col].astype(str)
            
            # Create primary label column
            primary_col = f"{col}_primary"
            df_copy[primary_col] = df_copy[col].apply(
                lambda x: str(x).split(delimiter)[0].strip() if pd.notna(x) else np.nan
            )
            
            # Create list column
            list_col = f"{col}_list"
            df_copy[list_col] = df_copy[col].apply(
                lambda x: [item.strip() for item in str(x).split(delimiter)] if pd.notna(x) else []
            )
    
    return df_copy

def merge_redundant_features(df, feature_groups):
    """
    Merge redundant features by selecting one canonical feature from each group.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        feature_groups (list): List of lists, where each inner list contains redundant features.
            The first feature in each list is considered the canonical one.
            
    Returns:
        pd.DataFrame: Dataframe with merged features.
    """
    df_copy = df.copy()
    
    # Default feature groups if not provided (example based on common redundancies)
    if not feature_groups:
        feature_groups = [
            ['screen_size_in', 'screen_diagonal', 'display_size'],
            ['ram_gb', 'memory_gb', 'system_memory'],
            ['storage_gb', 'hard_drive_gb', 'disk_space'],
            ['processor_speed', 'cpu_speed', 'cpu_frequency']
        ]
    
    cols_to_drop = []
    
    for group in feature_groups:
        if not group:
            continue
            
        canonical = group[0]
        redundant = group[1:]
        
        # Only process if the canonical feature exists
        if canonical not in df.columns:
            continue
        
        # For each redundant feature, if it exists, use it to fill NaNs in canonical
        for red_feat in redundant:
            if red_feat in df.columns:
                # Fill NaN values in canonical with values from redundant
                mask = df_copy[canonical].isna()
                df_copy.loc[mask, canonical] = df_copy.loc[mask, red_feat]
                cols_to_drop.append(red_feat)
    
    # Drop redundant columns
    df_copy = df_copy.drop(columns=cols_to_drop, errors='ignore')
    
    return df_copy

def handle_missing_data(df, numeric_strategy='median', categorical_strategy='Unknown', missing_threshold=0.3):
    """
    Handle missing data:
    - Flag columns with >30% missing as "missing_flag"
    - Impute others via median for numeric, "Unknown" for categoricals
    
    Args:
        df (pd.DataFrame): Input dataframe.
        numeric_strategy (str): Strategy for imputing numeric columns.
        categorical_strategy (str): Value to use for imputing categorical columns.
        missing_threshold (float): Threshold for flagging high-missing columns (0-1).
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values.
    """
    df_copy = df.copy()
    
    # Calculate missing percentage for each column
    missing_percentage = df.isna().mean()
    
    # Create missing flags for columns with high missing rates
    high_missing_cols = missing_percentage[missing_percentage > missing_threshold].index
    
    for col in high_missing_cols:
        flag_col = f"{col}_missing_flag"
        df_copy[flag_col] = df[col].isna().astype(int)
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Impute numeric columns with median
    for col in numeric_cols:
        if col in df.columns:
            df_copy[col] = df_copy[col].fillna(df[col].median() if not df[col].median() != df[col].median() else 0)
    
    # Impute categorical columns with "Unknown"
    for col in categorical_cols:
        if col in df.columns:
            df_copy[col] = df_copy[col].fillna(categorical_strategy)
    
    return df_copy

def clean_raw(df):
    """
    Clean raw dataframe by applying all preprocessing steps in sequence.
    
    Args:
        df (pd.DataFrame): Raw input dataframe.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # Define feature groups for merging
    feature_groups = [
        ['screen_size_in', 'screen_diagonal', 'display_size'],
        ['ram_gb', 'memory_gb', 'system_memory'],
        ['storage_gb', 'hard_drive_gb', 'disk_space'],
        ['processor_speed', 'cpu_speed', 'cpu_frequency']
    ]
    
    # Apply cleaning steps in sequence
    df = strip_text_from_numeric(df)
    df = parse_multi_label_fields(df)
    df = merge_redundant_features(df, feature_groups)
    df = handle_missing_data(df)
    
    return df

if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Loading raw data...")
    df_raw = load_data()
    print(f"Raw data shape: {df_raw.shape}")
    
    print("\nCleaning data...")
    df_clean = clean_raw(df_raw)
    print(f"Clean data shape: {df_clean.shape}")
    
    print("\nFirst 5 rows of cleaned data:")
    print(df_clean.head(5))
