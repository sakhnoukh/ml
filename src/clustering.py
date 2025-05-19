"""
Clustering Module for ML Marketplace

This module implements K-means clustering for computer configurations to identify
market segments and provide insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib
import os
from typing import Dict, List, Tuple, Any, Optional
import warnings

# Set up paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """
    Load and prepare data for clustering analysis.
    
    Returns:
        DataFrame with processed data ready for clustering
    """
    try:
        # Try to load the final processed dataset
        data_path = os.path.join(DATA_DIR, 'db_computers_final.csv')
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records from {data_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        # Fallback to training data if final dataset isn't available
        try:
            train_path = os.path.join(DATA_DIR, 'train_set.csv')
            df = pd.read_csv(train_path)
            print(f"Loaded {len(df)} records from {train_path}")
            return df
        except Exception as e2:
            print(f"Error loading training data: {e2}")
            raise Exception("Could not load any suitable dataset for clustering")

def prepare_data_for_clustering(df):
    """
    Prepare data for clustering by selecting and transforming relevant features.
    
    Args:
        df: Input DataFrame with computer data
        
    Returns:
        DataFrame with prepared features for clustering
    """
    # Select relevant hardware specification columns for clustering
    hardware_columns = [
        # CPU-related
        'cpu_brand', 'cpu_rating',
        # Memory
        'ram',
        # Storage
        'ssd_capacity',
        # Display
        'screen_size', 'screen_resolution',
        # Graphics
        'graphics_card',
        # Additional features
        'has_touchscreen', 'battery_life'
    ]
    
    # Create a copy to avoid SettingWithCopyWarning
    clustering_data = df.copy()
    
    # Check which columns exist in the DataFrame
    available_columns = [col for col in hardware_columns if col in df.columns]
    
    if not available_columns:
        # If none of the expected columns exist, check for Spanish column names
        # or use whatever numeric columns are available
        print("Expected columns not found. Using available numeric columns...")
        
        # Get numeric columns as fallback
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()[:5]  # Limit to 5 categorical
        
        # Remove price-related columns to avoid bias
        numeric_cols = [col for col in numeric_cols if 'precio' not in col.lower()]
        
        if not numeric_cols:
            raise Exception("No suitable numeric columns found for clustering")
        
        # Use the remaining columns
        available_columns = numeric_cols + categorical_cols
    
    # Filter to just the columns we need
    clustering_data = clustering_data[available_columns]
    
    # Handle categorical variables
    categorical_columns = clustering_data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        # One-hot encode categorical variables
        dummies = pd.get_dummies(clustering_data[col], prefix=col, drop_first=True)
        clustering_data = pd.concat([clustering_data, dummies], axis=1)
        clustering_data = clustering_data.drop(col, axis=1)
    
    # Fill missing values
    clustering_data = clustering_data.fillna(clustering_data.mean())
    
    return clustering_data

def determine_optimal_clusters(data, max_clusters=10):
    """
    Determine the optimal number of clusters using the elbow method and silhouette score.
    
    Args:
        data: Prepared data for clustering
        max_clusters: Maximum number of clusters to try
        
    Returns:
        Optimal number of clusters
    """
    print("Determining optimal number of clusters...")
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Calculate inertia (within-cluster sum of squares) for different k values
    inertia = []
    silhouette_scores = []
    
    # Try clustering with 2 to max_clusters clusters
    for k in range(2, max_clusters + 1):
        print(f"Testing {k} clusters...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)
        
        # Calculate silhouette score
        if k > 1:  # Silhouette score requires at least 2 clusters
            silhouette_avg = silhouette_score(scaled_data, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)
            print(f"Silhouette score for {k} clusters: {silhouette_avg:.4f}")
    
    # Plot the elbow curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), inertia, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'elbow_plot.png'))
    plt.close()
    
    # Find optimal number of clusters
    # Use silhouette score - higher is better
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    
    print(f"Optimal number of clusters: {optimal_clusters}")
    return optimal_clusters

def perform_clustering(data, n_clusters=5):
    """
    Perform K-means clustering on the prepared data.
    
    Args:
        data: Prepared data for clustering
        n_clusters: Number of clusters
        
    Returns:
        Tuple of (kmeans model, scaler, cluster labels)
    """
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Save the model and scaler
    joblib.dump(kmeans, os.path.join(MODEL_DIR, 'clustering_model.joblib'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'feature_scaler.joblib'))
    
    return kmeans, scaler, cluster_labels

def visualize_clusters(data, labels, n_clusters):
    """
    Visualize the clusters using PCA for dimensionality reduction.
    
    Args:
        data: Original data used for clustering
        labels: Cluster labels from K-means
        n_clusters: Number of clusters
    """
    print("Visualizing clusters...")
    
    # Apply PCA to reduce dimensions for visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)
    
    # Create DataFrame with principal components and cluster labels
    pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pc_df['Cluster'] = labels
    
    # Plot the clusters
    plt.figure(figsize=(12, 8))
    
    # Define color map
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Plot each cluster
    for i in range(n_clusters):
        cluster_data = pc_df[pc_df['Cluster'] == i]
        plt.scatter(
            cluster_data['PC1'], 
            cluster_data['PC2'],
            s=50, 
            c=[colors[i]],
            label=f'Cluster {i}'
        )
    
    # Add labels and legend
    plt.title('Computer Configuration Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Market Segments')
    plt.grid(alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_clusters(df, labels, n_clusters):
    """
    Analyze the characteristics of each cluster.
    
    Args:
        df: Original DataFrame with all features
        labels: Cluster labels from K-means
        n_clusters: Number of clusters
        
    Returns:
        DataFrame with cluster analysis
    """
    print("Analyzing cluster characteristics...")
    
    # Add cluster labels to original DataFrame
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = labels
    
    # Analyze each cluster
    cluster_analysis = []
    
    for i in range(n_clusters):
        cluster_df = df_with_clusters[df_with_clusters['Cluster'] == i]
        
        # Basic statistics
        cluster_size = len(cluster_df)
        cluster_percentage = (cluster_size / len(df)) * 100
        
        # Price analysis if price column exists
        price_stats = {}
        for col in df.columns:
            if 'precio' in col.lower() or 'price' in col.lower():
                if pd.api.types.is_numeric_dtype(df[col]):
                    price_stats = {
                        'mean_price': cluster_df[col].mean(),
                        'median_price': cluster_df[col].median(),
                        'min_price': cluster_df[col].min(),
                        'max_price': cluster_df[col].max()
                    }
                    break
        
        # Feature analysis
        feature_analysis = {}
        numeric_columns = df.select_dtypes(include=['number']).columns
        
        for col in numeric_columns:
            if col in cluster_df.columns and col != 'Cluster':
                feature_analysis[col] = {
                    'mean': cluster_df[col].mean(),
                    'median': cluster_df[col].median()
                }
        
        # Most common values for categorical features
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_analysis = {}
        
        for col in categorical_columns:
            if col in cluster_df.columns:
                value_counts = cluster_df[col].value_counts().head(3)
                categorical_analysis[col] = {
                    value: count for value, count in value_counts.items()
                }
        
        # Combine all analyses
        cluster_info = {
            'cluster_id': i,
            'size': cluster_size,
            'percentage': cluster_percentage,
            'price_stats': price_stats,
            'feature_stats': feature_analysis,
            'categorical_stats': categorical_analysis
        }
        
        cluster_analysis.append(cluster_info)
    
    # Convert to DataFrame for easier manipulation
    cluster_analysis_df = pd.DataFrame(cluster_analysis)
    
    # Optional: Save analysis to CSV
    cluster_analysis_df.to_csv(os.path.join(OUTPUT_DIR, 'cluster_analysis.csv'))
    
    return cluster_analysis_df

def prepare_config_for_clustering(config_dict):
    """
    Prepare a user configuration for cluster assignment.
    
    Args:
        config_dict: Dictionary with user's computer configuration
        
    Returns:
        DataFrame with prepared features in the same format as training data
    """
    try:
        # Load the scaler used during training
        scaler_path = os.path.join(MODEL_DIR, 'feature_scaler.joblib')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        
        scaler = joblib.load(scaler_path)
        
        # Extract features from config dictionary
        config_features = pd.DataFrame([config_dict])
        
        # Handle categorical variables (similar to the training process)
        categorical_columns = config_features.select_dtypes(include=['object']).columns
        
        # Load the model to get expected feature names
        model_path = os.path.join(MODEL_DIR, 'clustering_model.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        kmeans = joblib.load(model_path)
        
        # One-hot encode categorical variables based on training data categories
        # This is simplified and would need to match the exact encoding from training
        for col in categorical_columns:
            # Simple encoding for demonstration
            dummies = pd.get_dummies(config_features[col], prefix=col, drop_first=True)
            config_features = pd.concat([config_features, dummies], axis=1)
            config_features = config_features.drop(col, axis=1)
        
        # For older scikit-learn versions that don't have feature_names_in_
        # Just use the columns from the training data
        try:
            # Try to use feature_names_in_ if available (newer scikit-learn)
            if hasattr(kmeans, 'feature_names_in_'):
                expected_columns = kmeans.feature_names_in_
            else:
                # For older versions, we'll just use what we have
                print("Warning: KMeans model doesn't have feature_names_in_ attribute.")
                expected_columns = config_features.columns
                
            # Ensure all expected columns exist (adding missing ones with zeros)
            missing_cols = set(expected_columns) - set(config_features.columns)
            for col in missing_cols:
                config_features[col] = 0  # Will be imputed
                
            # Handle extra columns that weren't in training data
            extra_cols = set(config_features.columns) - set(expected_columns)
            if extra_cols:
                config_features = config_features.drop(columns=extra_cols)
                
            # If we have expected columns, reorder to match
            if len(expected_columns) > 0:
                # Make sure all columns in expected_columns exist
                for col in expected_columns:
                    if col not in config_features.columns:
                        config_features[col] = 0
                config_features = config_features[expected_columns]  
        except Exception as e:
            print(f"Warning during column matching: {e}")
            # Continue with what we have
        
        # Scale the features
        scaled_features = scaler.transform(config_features)
        
        return scaled_features
    
    except Exception as e:
        print(f"Error preparing config for clustering: {e}")
        # Return None or a default representation
        return None

def identify_cluster(config_dict):
    """
    Identify which cluster a given configuration belongs to.
    
    Args:
        config_dict: Dictionary with user's computer configuration
        
    Returns:
        Tuple of (cluster_id, confidence)
    """
    try:
        # Prepare the configuration for prediction
        prepared_config = prepare_config_for_clustering(config_dict)
        
        if prepared_config is None:
            return 0, 0.0  # Default cluster and zero confidence
        
        # Load the clustering model
        model_path = os.path.join(MODEL_DIR, 'clustering_model.joblib')
        kmeans = joblib.load(model_path)
        
        # Predict the cluster
        cluster_id = kmeans.predict(prepared_config)[0]
        
        # Calculate distance to cluster center as a confidence measure
        distances = kmeans.transform(prepared_config)
        min_distance = distances[0, cluster_id]
        max_distance = np.max(distances)
        
        # Normalize confidence (1 - normalized distance)
        confidence = 1.0 - (min_distance / max_distance if max_distance > 0 else 0)
        
        return cluster_id, confidence
    
    except Exception as e:
        print(f"Error identifying cluster: {e}")
        # Return default cluster and confidence
        return 0, 0.0

def run_clustering_analysis():
    """
    Run a complete clustering analysis workflow.
    
    Returns:
        DataFrame with cluster analysis
    """
    # Load data
    df = load_data()
    
    # Prepare data for clustering
    prepared_data = prepare_data_for_clustering(df)
    
    # Determine optimal number of clusters
    n_clusters = determine_optimal_clusters(prepared_data)
    
    # Perform clustering
    kmeans, scaler, labels = perform_clustering(prepared_data, n_clusters)
    
    # Visualize clusters
    visualize_clusters(prepared_data, labels, n_clusters)
    
    # Analyze clusters
    cluster_analysis = analyze_clusters(df, labels, n_clusters)
    
    print(f"Clustering analysis complete with {n_clusters} clusters")
    return cluster_analysis

if __name__ == "__main__":
    # Suppress common warnings during the clustering process
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    print("Starting clustering analysis...")
    cluster_analysis = run_clustering_analysis()
    print("Analysis complete and saved to outputs directory.")
