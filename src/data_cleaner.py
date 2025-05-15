#!/usr/bin/env python3
"""
Data Cleaning Module for ML Marketplace.

This module provides memory-efficient functions for cleaning numeric data,
including stripping text from numeric columns and unifying units.
"""

import os
import re
import pandas as pd
import numpy as np
import gc
from typing import List, Dict, Optional, Union, Tuple

# Unit conversion constants
UNIT_CONVERSIONS = {
    'tb_to_gb': 1024,  # 1 TB = 1024 GB
    'mb_to_gb': 1/1024,  # 1 MB = 1/1024 GB
    'kb_to_gb': 1/(1024*1024),  # 1 KB = 1/(1024*1024) GB
    'mhz_to_ghz': 1/1000,  # 1 MHz = 0.001 GHz
}

# Common patterns for different types of columns
PATTERNS = {
    'storage': r'(\d+\.?\d*|\d*\.\d+)\s*(TB|GB|MB|KB)',
    'processor': r'(\d+\.?\d*|\d*\.\d+)\s*(GHz|MHz)',
    'memory': r'(\d+\.?\d*|\d*\.\d+)\s*(GB|MB)',
    'screen': r'(\d+\.?\d*|\d*\.\d+)\s*(inches|"|in)',
    'generic': r'(\d+\.?\d*|\d*\.\d+)'
}

def detect_unit_columns(df_sample: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect columns that likely contain units based on patterns.
    
    Args:
        df_sample: Sample dataframe
        
    Returns:
        Dictionary mapping unit types to column names
    """
    column_types = {
        'storage': [],
        'processor': [],
        'memory': [],
        'screen': [],
        'other_numeric': []
    }
    
    # Check each column against patterns
    for col in df_sample.columns:
        if not pd.api.types.is_numeric_dtype(df_sample[col]):
            sample_values = df_sample[col].dropna().astype(str).tolist()[:20]
            
            # Skip if no samples or all values are NaN
            if not sample_values:
                continue
                
            # Check patterns
            if any(re.search(PATTERNS['storage'], str(val), re.IGNORECASE) for val in sample_values):
                column_types['storage'].append(col)
            elif any(re.search(PATTERNS['processor'], str(val), re.IGNORECASE) for val in sample_values):
                column_types['processor'].append(col)
            elif any(re.search(PATTERNS['memory'], str(val), re.IGNORECASE) for val in sample_values):
                column_types['memory'].append(col)
            elif any(re.search(PATTERNS['screen'], str(val), re.IGNORECASE) for val in sample_values):
                column_types['screen'].append(col)
            elif any(re.search(PATTERNS['generic'], str(val)) for val in sample_values):
                column_types['other_numeric'].append(col)
    
    return column_types

def clean_storage_value(val: str) -> float:
    """
    Extract numeric value from storage string and convert to GB.
    
    Args:
        val: String representing storage value
        
    Returns:
        Converted value in GB
    """
    if pd.isna(val):
        return np.nan
        
    val = str(val).lower()
    match = re.search(PATTERNS['storage'], val, re.IGNORECASE)
    
    if not match:
        # Try generic pattern if storage pattern fails
        match = re.search(PATTERNS['generic'], val)
        if not match:
            return np.nan
        else:
            return float(match.group(1))
    
    value = float(match.group(1))
    unit = match.group(2).lower()
    
    # Convert to GB
    if 'tb' in unit:
        return value * UNIT_CONVERSIONS['tb_to_gb']
    elif 'mb' in unit:
        return value * UNIT_CONVERSIONS['mb_to_gb']
    elif 'kb' in unit:
        return value * UNIT_CONVERSIONS['kb_to_gb']
    else:  # Already in GB
        return value

def clean_processor_value(val: str) -> float:
    """
    Extract numeric value from processor speed string and convert to GHz.
    
    Args:
        val: String representing processor speed
        
    Returns:
        Converted value in GHz
    """
    if pd.isna(val):
        return np.nan
        
    val = str(val).lower()
    match = re.search(PATTERNS['processor'], val, re.IGNORECASE)
    
    if not match:
        # Try generic pattern if processor pattern fails
        match = re.search(PATTERNS['generic'], val)
        if not match:
            return np.nan
        else:
            return float(match.group(1))
    
    value = float(match.group(1))
    unit = match.group(2).lower()
    
    # Convert to GHz
    if 'mhz' in unit:
        return value * UNIT_CONVERSIONS['mhz_to_ghz']
    else:  # Already in GHz
        return value

def clean_memory_value(val: str) -> float:
    """
    Extract numeric value from memory string and convert to GB.
    
    Args:
        val: String representing memory value
        
    Returns:
        Converted value in GB
    """
    if pd.isna(val):
        return np.nan
        
    val = str(val).lower()
    match = re.search(PATTERNS['memory'], val, re.IGNORECASE)
    
    if not match:
        # Try generic pattern if memory pattern fails
        match = re.search(PATTERNS['generic'], val)
        if not match:
            return np.nan
        else:
            return float(match.group(1))
    
    value = float(match.group(1))
    unit = match.group(2).lower()
    
    # Convert to GB
    if 'mb' in unit:
        return value * UNIT_CONVERSIONS['mb_to_gb']
    else:  # Already in GB
        return value

def clean_screen_value(val: str) -> float:
    """
    Extract numeric value from screen size string.
    
    Args:
        val: String representing screen size
        
    Returns:
        Numeric screen size value
    """
    if pd.isna(val):
        return np.nan
        
    val = str(val).lower()
    match = re.search(PATTERNS['screen'], val, re.IGNORECASE)
    
    if not match:
        # Try generic pattern if screen pattern fails
        match = re.search(PATTERNS['generic'], val)
        if not match:
            return np.nan
        else:
            return float(match.group(1))
    
    value = float(match.group(1))
    return value

def clean_generic_numeric(val: str) -> float:
    """
    Extract numeric value from a string.
    
    Args:
        val: String potentially containing a numeric value
        
    Returns:
        Extracted numeric value
    """
    if pd.isna(val):
        return np.nan
        
    val = str(val)
    match = re.search(PATTERNS['generic'], val)
    
    if not match:
        return np.nan
    
    return float(match.group(1))

def process_chunk(chunk: pd.DataFrame, column_types: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Process a chunk of data by cleaning and unifying units.
    
    Args:
        chunk: DataFrame chunk to process
        column_types: Dictionary mapping column types to column names
        
    Returns:
        Processed DataFrame chunk
    """
    # Create a copy to avoid modifying the original
    chunk_copy = chunk.copy()
    
    # Process each column type
    for col in column_types['storage']:
        if col in chunk_copy.columns:
            chunk_copy[col] = chunk_copy[col].apply(clean_storage_value)
    
    for col in column_types['processor']:
        if col in chunk_copy.columns:
            chunk_copy[col] = chunk_copy[col].apply(clean_processor_value)
    
    for col in column_types['memory']:
        if col in chunk_copy.columns:
            chunk_copy[col] = chunk_copy[col].apply(clean_memory_value)
    
    for col in column_types['screen']:
        if col in chunk_copy.columns:
            chunk_copy[col] = chunk_copy[col].apply(clean_screen_value)
    
    for col in column_types['other_numeric']:
        if col in chunk_copy.columns:
            chunk_copy[col] = chunk_copy[col].apply(clean_generic_numeric)
    
    return chunk_copy

def clean_data_in_chunks(input_file: str, 
                        output_file: str,
                        column_types: Optional[Dict[str, List[str]]] = None,
                        chunk_size: int = 1000) -> None:
    """
    Clean data in chunks to avoid memory explosion.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        column_types: Dictionary mapping column types to column names (if None, will be detected)
        chunk_size: Size of chunks to process
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # If column types not provided, detect them from a sample
    if column_types is None:
        df_sample = pd.read_csv(input_file, nrows=500, encoding='utf-8-sig')
        column_types = detect_unit_columns(df_sample)
        del df_sample
        gc.collect()
    
    # Display column types
    print("Column types detected:")
    for col_type, cols in column_types.items():
        print(f"  {col_type}: {len(cols)} columns")
        if cols:
            print(f"    Example: {cols[0]}")
    
    # Initialize variables
    first_chunk = True
    
    # Process in chunks
    for chunk_num, chunk in enumerate(pd.read_csv(input_file, 
                                                chunksize=chunk_size, 
                                                encoding='utf-8-sig')):
        print(f"Processing chunk {chunk_num+1}...")
        
        # Process chunk
        processed_chunk = process_chunk(chunk, column_types)
        
        # Write to output file
        if first_chunk:
            processed_chunk.to_csv(output_file, index=False, mode='w')
            first_chunk = False
        else:
            processed_chunk.to_csv(output_file, index=False, mode='a', header=False)
        
        # Free memory
        del chunk
        del processed_chunk
        gc.collect()
    
    print(f"Data cleaning complete. Cleaned data saved to {output_file}")

def clean_dataset(input_file: str, output_file: str) -> None:
    """
    Clean the entire dataset.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    print(f"Starting data cleaning for {input_file}...")
    
    # Detect column types from a sample
    df_sample = pd.read_csv(input_file, nrows=500, encoding='utf-8-sig')
    column_types = detect_unit_columns(df_sample)
    
    # Free memory
    del df_sample
    gc.collect()
    
    # Clean data in chunks
    clean_data_in_chunks(
        input_file=input_file,
        output_file=output_file,
        column_types=column_types,
        chunk_size=1000
    )

if __name__ == "__main__":
    # Example usage
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    input_file = os.path.join(data_dir, "db_computers_2025_raw.csv")
    output_file = os.path.join(data_dir, "db_computers_cleaned_units.csv")
    
    # Clean the dataset
    clean_dataset(input_file, output_file)
