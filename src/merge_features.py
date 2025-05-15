#!/usr/bin/env python3
"""
Feature Merger for ML Marketplace.

This module identifies and merges redundant features in the computer dataset.
"""

import os
import pandas as pd
import numpy as np
import gc
from typing import List, Dict, Tuple, Optional

# Define common feature groups that are likely redundant
COMMON_FEATURE_GROUPS = [
    # Screen size features
    ['Pantalla_Tamaño de la pantalla', 'Pantalla_Diagonal de la pantalla', 'Pantalla_Resolución'], 
    
    # Storage features
    ['Disco duro_Capacidad total de almacenamiento', 'Disco duro_Capacidad HDD', 'Disco duro_Capacidad SSD'],
    
    # RAM/Memory features
    ['RAM_Memoria interna', 'Memoria instalada', 'RAM_Tamaño de memoria'],
    
    # Processor features
    ['Procesador_Frecuencia del procesador', 'Procesador_Velocidad del procesador', 'Procesador_Velocidad de reloj'],
    
    # GPU features
    ['Gráfica_Tarjeta gráfica', 'Gráfica_Modelo de adaptador gráfico', 'Gráfica_Familia de adaptador de gráficos'],
    
    # Battery features
    ['Alimentación_Número de celdas', 'Batería_Celdas de la batería'],
    
    # OS features
    ['Sistema operativo', 'SO instalado', 'OS_Nombre']
]

def detect_column_correlations(df: pd.DataFrame, 
                              sample_size: int = 5000, 
                              correlation_threshold: float = 0.7) -> List[List[str]]:
    """
    Detect correlated numeric columns to identify potentially redundant features.
    
    Args:
        df: Input dataframe or a sample
        sample_size: Number of rows to sample for correlation analysis
        correlation_threshold: Threshold for considering columns as correlated
        
    Returns:
        List of lists, where each inner list contains correlated column names
    """
    # Sample the dataframe to make computation faster
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df
    
    # Get numeric columns
    numeric_cols = df_sample.select_dtypes(include=['number']).columns.tolist()
    
    # Calculate correlation matrix
    try:
        corr_matrix = df_sample[numeric_cols].corr().abs().fillna(0)
    except Exception as e:
        print(f"Error calculating correlation matrix: {e}")
        return []
    
    # Initialize groups
    correlated_groups = []
    
    # Find groups of correlated features
    visited = set()
    
    for col in corr_matrix.columns:
        if col in visited:
            continue
            
        # Find columns highly correlated with this one
        correlated = corr_matrix.index[corr_matrix[col] > correlation_threshold].tolist()
        
        # If we found at least one other correlated column
        if len(correlated) > 1:  # > 1 because a column is perfectly correlated with itself
            # Add to our groups and mark as visited
            group = sorted(correlated)
            correlated_groups.append(group)
            visited.update(group)
    
    return correlated_groups

def detect_similar_column_names(df: pd.DataFrame) -> List[List[str]]:
    """
    Detect columns with similar names that might be redundant.
    
    Args:
        df: Input dataframe
        
    Returns:
        List of lists, where each inner list contains similar column names
    """
    from collections import defaultdict
    
    # Group columns by common keywords
    keyword_groups = defaultdict(list)
    keywords = ['screen', 'size', 'resolution', 'diagonal', 'storage', 'memory', 'ram', 
                'processor', 'cpu', 'graphic', 'gpu', 'battery', 'os', 'weight', 'price']
    
    for col in df.columns:
        col_lower = col.lower()
        for keyword in keywords:
            if keyword in col_lower:
                keyword_groups[keyword].append(col)
    
    # Filter groups to those with at least 2 columns
    similar_groups = [group for group in keyword_groups.values() if len(group) >= 2]
    
    return similar_groups

def select_canonical_feature(df: pd.DataFrame, group: List[str]) -> str:
    """
    Select the canonical feature from a group of redundant features.
    
    Args:
        df: Input dataframe
        group: Group of redundant feature names
        
    Returns:
        Name of the canonical feature
    """
    # Strategy: select the feature with the least missing values
    missing_counts = {col: df[col].isna().sum() for col in group if col in df.columns}
    
    if not missing_counts:
        return None  # No valid columns found
        
    # Select column with minimum missing values
    canonical = min(missing_counts.items(), key=lambda x: x[1])[0]
    return canonical

def merge_redundant_features(input_file: str, output_file: str, chunk_size: int = 1000) -> Dict[str, str]:
    """
    Merge redundant features in the dataset.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        chunk_size: Size of chunks to process
        
    Returns:
        Dictionary mapping canonical features to lists of redundant features
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # First pass: read a sample to identify redundant feature groups
    print("Loading sample to identify redundant features...")
    df_sample = pd.read_csv(input_file, nrows=5000, encoding='utf-8-sig')
    
    # Combine predefined groups with detected ones
    feature_groups = COMMON_FEATURE_GROUPS.copy()
    
    # Filter groups to only include columns that exist in the dataset
    filtered_groups = []
    for group in feature_groups:
        valid_group = [col for col in group if col in df_sample.columns]
        if len(valid_group) >= 2:
            filtered_groups.append(valid_group)
    
    # Try to detect additional correlated columns
    correlated_groups = detect_column_correlations(df_sample)
    filtered_groups.extend(correlated_groups)
    
    # Get similar column name groups
    similar_name_groups = detect_similar_column_names(df_sample)
    filtered_groups.extend(similar_name_groups)
    
    # Select canonical features for each group
    canonical_mapping = {}
    for group in filtered_groups:
        canonical = select_canonical_feature(df_sample, group)
        if canonical:
            redundant = [col for col in group if col != canonical and col in df_sample.columns]
            if redundant:
                canonical_mapping[canonical] = redundant
    
    # Free memory
    del df_sample
    gc.collect()
    
    print(f"Identified {len(canonical_mapping)} canonical features with redundant columns")
    
    # Display the canonical mapping
    for canonical, redundant in canonical_mapping.items():
        print(f"  Canonical: {canonical}")
        print(f"  Redundant: {', '.join(redundant)}")
    
    # Second pass: process data in chunks and merge redundant features
    print("\nMerging redundant features in chunks...")
    
    # Initialize variables
    first_chunk = True
    
    # Process chunks
    for chunk_num, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size, encoding='utf-8-sig')):
        print(f"Processing chunk {chunk_num+1}...")
        
        # Merge redundant features
        for canonical, redundant in canonical_mapping.items():
            if canonical in chunk.columns:
                for red_col in redundant:
                    if red_col in chunk.columns:
                        # Fill missing values in canonical with values from redundant
                        mask = chunk[canonical].isna()
                        chunk.loc[mask, canonical] = chunk.loc[mask, red_col]
                
                # After merging, drop redundant columns
                chunk_cols = [col for col in redundant if col in chunk.columns]
                chunk = chunk.drop(columns=chunk_cols, errors='ignore')
        
        # Write to output file
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        chunk.to_csv(output_file, index=False, mode=mode, header=header)
        
        if first_chunk:
            first_chunk = False
        
        # Free memory
        del chunk
        gc.collect()
    
    print(f"Redundant feature merging complete. Result saved to {output_file}")
    return canonical_mapping

if __name__ == "__main__":
    # Example usage
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    # Use the cleaned data file (with handled missing values) as input
    input_file = os.path.join(data_dir, "db_computers_final.csv")
    output_file = os.path.join(data_dir, "db_computers_merged.csv")
    
    # Merge redundant features
    canonical_mapping = merge_redundant_features(input_file, output_file)
