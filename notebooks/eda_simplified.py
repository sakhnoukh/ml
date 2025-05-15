#!/usr/bin/env python3
"""
Memory-Efficient EDA script for ML Marketplace project.
This script processes data in chunks to avoid memory issues.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'data', 'db_computers_2025_raw.csv')
outputs_dir = os.path.join(project_dir, 'outputs')
os.makedirs(outputs_dir, exist_ok=True)

# Set up plotting
plt.style.use('default')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.max_open_warning'] = 50  # Avoid too many figures warning

print("Starting EDA analysis...")
print(f"Output will be saved to: {outputs_dir}")

# Define unit conversion functions (simplified from preprocessing.py)
def extract_numeric(val):
    """Extract numeric value from string, handling units."""
    if pd.isna(val):
        return np.nan
    
    val = str(val).lower()
    
    # Try to extract a number
    import re
    num_match = re.search(r'(\d+\.?\d*|\d*\.\d+)', val)
    if not num_match:
        return np.nan
    
    num_val = float(num_match.group(1))
    
    # Handle units
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
    else:
        return num_val

# Read data in chunks
print("\nLoading and processing data in chunks...")
chunk_size = 500  # Process 500 rows at a time

# Step 1: Get column information first
first_chunk = pd.read_csv(data_path, encoding='utf-8-sig', nrows=5)
all_columns = first_chunk.columns.tolist()
numeric_candidates = []
categorical_columns = []

# Identify likely numeric columns containing units
for col in all_columns:
    # Check if the column might contain unit values
    sample_val = first_chunk[col].iloc[0] if not first_chunk[col].isna().all() else None
    if sample_val and isinstance(sample_val, str):
        if any(unit in str(sample_val).lower() for unit in ['gb', 'tb', 'mb', 'ghz', 'mhz', 'inches', 'in', '"']):
            numeric_candidates.append(col)
        else:
            categorical_columns.append(col)
    elif pd.api.types.is_numeric_dtype(first_chunk[col]):
        numeric_candidates.append(col)
    else:
        categorical_columns.append(col)

del first_chunk
print(f"Found {len(numeric_candidates)} potential numeric columns and {len(categorical_columns)} categorical columns")

# Initialize DataFrames for analysis
print("\nPreprocessing numeric columns...")
numeric_processed = pd.DataFrame()

# Read and process chunks
for chunk_num, chunk in enumerate(pd.read_csv(data_path, encoding='utf-8-sig', chunksize=chunk_size)):
    print(f"Processing chunk {chunk_num+1}...")
    
    # Convert potential numeric columns
    for col in numeric_candidates:
        if col in chunk.columns:
            # Only convert if not already numeric
            if not pd.api.types.is_numeric_dtype(chunk[col]):
                chunk[col] = chunk[col].apply(extract_numeric)
    
    # Keep only the processed numeric data for analysis
    chunk_numeric = chunk[numeric_candidates].copy()
    
    # Append to our numeric dataset (limited to 1000 rows for memory efficiency)
    if len(numeric_processed) < 1000:
        numeric_processed = pd.concat([numeric_processed, chunk_numeric], ignore_index=True)
    
    # Free memory
    del chunk_numeric
    del chunk
    gc.collect()

# Ensure we have clean numeric data
numeric_processed = numeric_processed.apply(pd.to_numeric, errors='coerce')
print(f"Processed {len(numeric_processed)} rows for numerical analysis")

# 2. Generate summary statistics
print("\nGenerating summary statistics...")
try:
    # Generate summary stats for numeric columns
    stats = numeric_processed.describe().T
    stats['missing%'] = numeric_processed.isna().mean() * 100
    stats = stats.sort_values('missing%', ascending=False)
    
    # Save to CSV
    stats_file = os.path.join(outputs_dir, 'summary_stats.csv')
    stats.to_csv(stats_file)
    print(f"Summary statistics saved to {stats_file}")
    
    # Generate a text summary of the most important columns
    with open(os.path.join(outputs_dir, 'data_summary.txt'), 'w') as f:
        f.write("COMPUTER DATASET SUMMARY\n")
        f.write(f"Total columns analyzed: {len(numeric_processed.columns)}\n\n")
        
        # Top 10 columns by completeness
        top_cols = stats.sort_values('missing%').head(10).index.tolist()
        f.write("TOP 10 MOST COMPLETE NUMERIC COLUMNS:\n")
        for col in top_cols:
            missing = stats.loc[col, 'missing%']
            mean = stats.loc[col, 'mean']
            f.write(f"  {col}: {missing:.1f}% missing, mean={mean:.2f}\n")
    
    print("Text summary saved")
except Exception as e:
    print(f"Error in summary statistics: {e}")
    
# 3. Correlation matrix (memory-optimized)
print("\nGenerating correlation matrix...")
try:
    # Filter columns to those with sufficient non-null values
    valid_cols = [col for col in numeric_processed.columns 
                  if numeric_processed[col].notna().sum() > len(numeric_processed) * 0.5]
    
    # If we have too many columns, keep only top 20 by variance
    if len(valid_cols) > 20:
        var_series = numeric_processed[valid_cols].var()
        valid_cols = var_series.nlargest(20).index.tolist()
    
    print(f"Using {len(valid_cols)} columns for correlation analysis")
    
    if len(valid_cols) > 1:
        # Calculate correlation and fill NaN with 0
        corr_matrix = numeric_processed[valid_cols].corr().fillna(0)
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, 
                    fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Numeric Features', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, 'correlation_matrix.png'), dpi=150)
        plt.close()
        print("Correlation matrix saved")
    else:
        print("Not enough valid numeric columns for correlation analysis")
except Exception as e:
    print(f"Error in correlation matrix: {e}")

# 4. Distribution plots
print("\nGenerating distribution plots...")
try:
    # Get top 5 numeric features by variance (that have sufficient non-null values)
    var_cols = [col for col in numeric_processed.columns 
                if numeric_processed[col].notna().sum() > 100]
    
    if len(var_cols) > 0:
        var_series = numeric_processed[var_cols].var()
        top_features = var_series.nlargest(min(5, len(var_series))).index.tolist()
        
        for feature in top_features:
            data = numeric_processed[feature].dropna()
            
            # Filter out extreme outliers for better visualization
            q1, q3 = data.quantile(0.05), data.quantile(0.95)
            filtered_data = data[(data >= q1) & (data <= q3)]
            
            plt.figure(figsize=(10, 4))
            
            # Histogram
            plt.subplot(1, 2, 1)
            sns.histplot(filtered_data, kde=True)
            plt.title(f'Histogram: {feature}')
            
            # Boxplot
            plt.subplot(1, 2, 2)
            sns.boxplot(x=filtered_data)
            plt.title(f'Boxplot: {feature}')
            
            plt.tight_layout()
            plt.savefig(os.path.join(outputs_dir, f'dist_{feature}.png'), dpi=150)
            plt.close()
        
        print(f"Created distribution plots for {len(top_features)} features")
    else:
        print("Not enough data for distribution plots")
except Exception as e:
    print(f"Error in distribution plots: {e}")

# 5. Categorical analysis
print("\nAnalyzing categorical columns...")
try:
    # Process categorical data in chunks
    categorical_data = {}
    
    # Read chunks and count category frequencies
    for chunk_num, chunk in enumerate(pd.read_csv(data_path, encoding='utf-8-sig', 
                                                 usecols=categorical_columns[:10],  # Limit to 10 categorical cols
                                                 chunksize=500)):
        for col in chunk.columns:
            if col not in categorical_data:
                categorical_data[col] = {}
                
            # Count values
            for val in chunk[col].dropna().values:
                val_str = str(val)
                if val_str in categorical_data[col]:
                    categorical_data[col][val_str] += 1
                else:
                    categorical_data[col][val_str] = 1
                    
        print(f"Processed chunk {chunk_num+1} for categorical analysis")
    
    # Generate bar plots for top categories
    for col, counts in categorical_data.items():
        # Convert to Series and get top values
        count_series = pd.Series(counts).sort_values(ascending=False)
        top_values = count_series.head(10)
        
        # Create bar plot
        plt.figure(figsize=(10, 5))
        sns.barplot(x=top_values.index, y=top_values.values)
        plt.title(f'Top Values: {col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, f'cat_{col}.png'), dpi=150)
        plt.close()
    
    print(f"Created categorical plots for {len(categorical_data)} columns")
except Exception as e:
    print(f"Error in categorical analysis: {e}")

# 6. Simple clustering
print("\nPerforming basic cluster analysis...")
try:
    # Get 2-3 of the most complete numeric columns for clustering
    cluster_cols = [col for col in numeric_processed.columns
                   if numeric_processed[col].notna().sum() > numeric_processed.shape[0] * 0.8][:3]
    
    if len(cluster_cols) >= 2:
        cluster_data = numeric_processed[cluster_cols].dropna()
        
        if len(cluster_data) >= 100:
            print(f"Using {len(cluster_data)} samples with features: {cluster_cols}")
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Try a few k values
            k_values = [2, 3, 4]
            scores = []
            
            for k in k_values:
                # Train K-means with low iterations to save memory
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
                labels = kmeans.fit_predict(scaled_data)
                
                # Calculate silhouette score
                try:
                    score = silhouette_score(scaled_data, labels)
                    scores.append(score)
                    print(f"K={k}, Silhouette score: {score:.3f}")
                except Exception as e:
                    print(f"Error calculating score for k={k}: {e}")
                    scores.append(-1)
            
            # Plot scores if any are valid
            if any(s > 0 for s in scores):
                plt.figure(figsize=(8, 5))
                plt.plot(k_values, scores, 'o-')
                plt.xlabel('Number of Clusters (k)')
                plt.ylabel('Silhouette Score')
                plt.title('Clustering Performance')
                plt.grid(True)
                plt.savefig(os.path.join(outputs_dir, 'clustering.png'), dpi=150)
                plt.close()
                print("Saved clustering analysis")
        else:
            print("Not enough complete data rows for clustering")
    else:
        print("Not enough numeric columns for clustering")
except Exception as e:
    print(f"Error in clustering: {e}")

print(f"\nEDA analysis complete! All outputs saved to: {outputs_dir}")
