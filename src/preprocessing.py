"""
Data preprocessing module for the ML Marketplace.

This module handles the loading, cleaning, and transformation of the raw computer dataset.
It focuses on cleaning numeric fields, handling missing values, and preparing data for
feature engineering.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the raw computer dataset with appropriate encoding.
    
    Args:
        file_path: Path to the raw CSV file
        
    Returns:
        DataFrame containing the raw data
    """
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)
        logger.info(f"Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def clean_numeric_with_units(df: pd.DataFrame, column: str, 
                             unit_patterns: Dict[str, float] = None) -> pd.DataFrame:
    """
    Clean numeric columns by stripping text and converting units.
    
    Args:
        df: DataFrame containing the data
        column: Column name to clean
        unit_patterns: Dictionary mapping unit patterns to multipliers
        
    Returns:
        DataFrame with cleaned column
    """
    if unit_patterns is None:
        # Default unit patterns for computer specs
        unit_patterns = {
            r'TB': 1.0,           # Terabytes
            r'GB': 1/1024,        # Convert GB to TB
            r'MB': 1/(1024*1024), # Convert MB to TB
            r'GHz': 1.0,          # GHz
            r'MHz': 1/1000,       # Convert MHz to GHz
            r'"': 1.0,            # Inches
            r'inch|inches': 1.0   # Inches
        }
    
    df_copy = df.copy()
    
    if column not in df_copy.columns:
        logger.warning(f"Column {column} not found in DataFrame")
        return df_copy
    
    logger.info(f"Cleaning numeric column: {column}")
    
    # Convert to string to ensure consistent handling
    df_copy[column] = df_copy[column].astype(str)
    
    # Function to extract numeric value and apply unit conversion
    def extract_and_convert(value):
        if pd.isna(value) or value == 'nan':
            return np.nan
        
        value = str(value).strip()
        
        # Find the first unit pattern that matches
        for pattern, multiplier in unit_patterns.items():
            match = re.search(f'({pattern})', value, re.IGNORECASE)
            if match:
                # Extract numeric part and convert
                numeric_part = re.sub(r'[^0-9.]', '', value.split(match.group(1))[0])
                try:
                    return float(numeric_part) * multiplier
                except ValueError:
                    return np.nan
        
        # If no unit patterns match, just extract numeric value
        numeric_part = re.sub(r'[^0-9.]', '', value)
        try:
            return float(numeric_part)
        except ValueError:
            return np.nan
    
    # Apply the extraction and conversion
    df_copy[column] = df_copy[column].apply(extract_and_convert)
    
    return df_copy


def merge_duplicate_features(df: pd.DataFrame, feature_groups: List[List[str]]) -> pd.DataFrame:
    """
    Merge duplicate features into a single feature.
    
    Args:
        df: DataFrame containing the data
        feature_groups: List of lists where each inner list contains columns to be merged
        
    Returns:
        DataFrame with merged features
    """
    df_copy = df.copy()
    
    for group in feature_groups:
        if not all(col in df_copy.columns for col in group):
            missing = [col for col in group if col not in df_copy.columns]
            logger.warning(f"Columns {missing} not found in DataFrame, skipping group {group}")
            continue
            
        new_col_name = group[0]  # Use first column name as the merged column name
        logger.info(f"Merging duplicate features: {group} -> {new_col_name}")
        
        # For each row, take the first non-NaN value from the group
        df_copy[new_col_name] = df_copy[group].apply(
            lambda row: next((val for val in row if not pd.isna(val)), np.nan), 
            axis=1
        )
        
        # Drop the other columns in the group (except the first one which is kept)
        if len(group) > 1:
            df_copy = df_copy.drop(columns=group[1:])
    
    return df_copy


def handle_missing_values(df: pd.DataFrame, strategy: Dict[str, str] = None) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: DataFrame containing the data
        strategy: Dictionary mapping column names to imputation strategies
            Supported strategies: 'mean', 'median', 'mode', 'constant:value',
            'flag' (adds a flag column indicating missingness)
        
    Returns:
        DataFrame with handled missing values
    """
    if strategy is None:
        strategy = {}
    
    df_copy = df.copy()
    
    # If no specific strategies provided, use sensible defaults
    for col in df_copy.columns:
        if col not in strategy:
            if df_copy[col].dtype.kind in 'ifc':  # integer, float, complex
                strategy[col] = 'median'
            else:
                strategy[col] = 'mode'
    
    logger.info("Handling missing values")
    
    for col, method in strategy.items():
        if col not in df_copy.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue
            
        missing_mask = df_copy[col].isna()
        missing_count = missing_mask.sum()
        
        if missing_count == 0:
            logger.info(f"No missing values in column {col}")
            continue
            
        logger.info(f"Column {col}: {missing_count} missing values, strategy: {method}")
        
        if method == 'mean':
            if df_copy[col].dtype.kind in 'ifc':
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
            else:
                logger.warning(f"Mean imputation not suitable for non-numeric column {col}, using mode instead")
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
                
        elif method == 'median':
            if df_copy[col].dtype.kind in 'ifc':
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            else:
                logger.warning(f"Median imputation not suitable for non-numeric column {col}, using mode instead")
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
                
        elif method == 'mode':
            if not df_copy[col].mode().empty:
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
                
        elif method.startswith('constant:'):
            constant_value = method.split(':', 1)[1]
            df_copy[col] = df_copy[col].fillna(constant_value)
            
        elif method == 'flag':
            # Create a flag column indicating missingness
            flag_col_name = f"{col}_missing"
            df_copy[flag_col_name] = missing_mask.astype(int)
            
        else:
            logger.warning(f"Unknown imputation strategy {method} for column {col}")
    
    return df_copy


def preprocess_data(
    file_path: str,
    numeric_columns: List[str] = None,
    feature_groups: List[List[str]] = None,
    imputation_strategy: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for the computer dataset.
    
    Args:
        file_path: Path to the raw CSV file
        numeric_columns: List of numeric columns to clean
        feature_groups: List of lists where each inner list contains columns to be merged
        imputation_strategy: Dictionary mapping column names to imputation strategies
        
    Returns:
        Preprocessed DataFrame ready for feature engineering
    """
    # Load the data
    df = load_data(file_path)
    
    # Default numeric columns if none provided
    if numeric_columns is None:
        numeric_columns = []
    
    # Clean numeric columns
    for col in numeric_columns:
        df = clean_numeric_with_units(df, col)
    
    # Merge duplicate features
    if feature_groups is not None:
        df = merge_duplicate_features(df, feature_groups)
    
    # Handle missing values
    df = handle_missing_values(df, imputation_strategy)
    
    logger.info(f"Preprocessing complete. Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


if __name__ == "__main__":
    # Example usage
    # df = preprocess_data(
    #     file_path="../data/db_computers_2025_raw.csv",
    #     numeric_columns=["Storage", "RAM", "Screen Size", "Processor Speed"],
    #     feature_groups=[["Screen Size", "Display Size"]],
    #     imputation_strategy={"Price": "median", "Brand": "mode"}
    # )
    pass
