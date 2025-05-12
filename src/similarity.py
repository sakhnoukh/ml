"""
Similarity search module for the ML Marketplace.

This module implements similarity search functionality for finding computer systems
with similar specifications. It uses k-NN or cosine similarity measures on feature vectors.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimilaritySearch:
    """
    Class for performing similarity search on computer specifications.
    """
    
    def __init__(self, feature_names: List[str], metric: str = 'euclidean'):
        """
        Initialize the similarity search model.
        
        Args:
            feature_names: Names of features used for similarity
            metric: Distance metric ('euclidean' or 'cosine')
        """
        self.feature_names = feature_names
        self.metric = metric
        self.model = None
        self.scaler = StandardScaler()
        self.reference_data = None
        self.reference_features = None
        self.original_data = None
        
    def fit(self, X: np.ndarray, original_data: pd.DataFrame):
        """
        Fit the similarity model on feature vectors.
        
        Args:
            X: Feature matrix
            original_data: Original DataFrame with all columns
        """
        logger.info(f"Fitting similarity model with {self.metric} metric")
        
        # Save reference data
        self.reference_features = X
        self.original_data = original_data
        
        # Scale features
        self.reference_features = self.scaler.fit_transform(self.reference_features)
        
        # Use k-NN for euclidean distance
        if self.metric == 'euclidean':
            self.model = NearestNeighbors(
                n_neighbors=min(10, len(self.reference_features)),
                metric='euclidean'
            )
            self.model.fit(self.reference_features)
            
    def find_similar_items(self, query_features: np.ndarray, k: int = 5) -> pd.DataFrame:
        """
        Find k most similar items to the query.
        
        Args:
            query_features: Query feature vector
            k: Number of similar items to return
            
        Returns:
            DataFrame with similar items, including distance and price comparison
        """
        logger.info(f"Finding {k} similar items")
        
        # Scale query features
        query_features_scaled = self.scaler.transform(query_features.reshape(1, -1))
        
        if self.metric == 'euclidean':
            # Get k+1 nearest neighbors (first one is the query itself if it exists in the reference data)
            distances, indices = self.model.kneighbors(
                query_features_scaled, 
                n_neighbors=min(k+1, len(self.reference_features))
            )
            # Use the first k results (excluding the query if it's in the reference data)
            distances = distances[0]
            indices = indices[0]
            
        else:  # Cosine similarity
            # Calculate cosine similarity
            similarity_scores = cosine_similarity(
                query_features_scaled, 
                self.reference_features
            )[0]
            
            # Convert to distances (1 - similarity)
            distances = 1 - similarity_scores
            
            # Get top k indices
            indices = np.argsort(distances)[:k+1]
            distances = distances[indices]
        
        # Create results DataFrame
        results = []
        
        for idx, dist in zip(indices, distances):
            # Original data row
            row = self.original_data.iloc[idx].copy()
            # Add distance
            row['similarity_distance'] = dist
            # Add to results
            results.append(row)
            
        similar_items = pd.DataFrame(results)
        
        # If 'Price' column exists, calculate price difference from query
        if 'Price' in similar_items.columns and 'Price' in query_features.index:
            query_price = query_features['Price']
            similar_items['price_difference'] = similar_items['Price'] - query_price
            similar_items['price_difference_pct'] = (similar_items['price_difference'] / query_price) * 100
            
        return similar_items
    
    def get_feature_distances(self, query_features: pd.Series, similar_item: pd.Series) -> Dict[str, float]:
        """
        Calculate feature-wise distances between query and a similar item.
        
        Args:
            query_features: Query feature Series
            similar_item: Similar item Series
            
        Returns:
            Dictionary of feature distances
        """
        feature_distances = {}
        
        # Calculate distances for each feature
        for feature in self.feature_names:
            if feature in query_features and feature in similar_item:
                if pd.api.types.is_numeric_dtype(query_features[feature]):
                    # For numeric features, calculate absolute difference
                    feature_distances[feature] = abs(query_features[feature] - similar_item[feature])
                else:
                    # For categorical features, 0 if same, 1 if different
                    feature_distances[feature] = 0 if query_features[feature] == similar_item[feature] else 1
        
        return feature_distances
    
    def normalize_feature_distances(self, feature_distances: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize feature distances to [0, 1] scale.
        
        Args:
            feature_distances: Dictionary of feature distances
            
        Returns:
            Dictionary of normalized feature distances
        """
        # Find maximum value for each feature in the reference data
        feature_max = {}
        for feature in feature_distances.keys():
            if feature in self.original_data.columns and pd.api.types.is_numeric_dtype(self.original_data[feature]):
                feature_max[feature] = self.original_data[feature].max() - self.original_data[feature].min()
            else:
                feature_max[feature] = 1  # For categorical features
        
        # Normalize
        normalized_distances = {}
        for feature, distance in feature_distances.items():
            if feature_max[feature] > 0:
                normalized_distances[feature] = distance / feature_max[feature]
            else:
                normalized_distances[feature] = 0
        
        return normalized_distances
    
    def plot_feature_comparison(self, query_features: pd.Series, similar_item: pd.Series, 
                              plot_type: str = 'radar', output_path: Optional[str] = None):
        """
        Create a visualization comparing query features with a similar item.
        
        Args:
            query_features: Query feature Series
            similar_item: Similar item Series
            plot_type: Type of plot ('radar' or 'parallel')
            output_path: Path to save the plot
        """
        # Get feature distances
        feature_distances = self.get_feature_distances(query_features, similar_item)
        
        # Normalize distances
        normalized_distances = self.normalize_feature_distances(feature_distances)
        
        # Select common numeric features
        common_features = [
            f for f in self.feature_names 
            if f in query_features and f in similar_item and 
            pd.api.types.is_numeric_dtype(query_features[f])
        ]
        
        if plot_type == 'radar':
            # Create radar plot
            plt.figure(figsize=(10, 8))
            
            # Set up the radar plot
            angles = np.linspace(0, 2*np.pi, len(common_features), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))
            
            # Plot query features
            query_values = [query_features[f] for f in common_features]
            query_values = np.concatenate((query_values, [query_values[0]]))
            plt.polar(angles, query_values, 'o-', label='Query')
            
            # Plot similar item features
            similar_values = [similar_item[f] for f in common_features]
            similar_values = np.concatenate((similar_values, [similar_values[0]]))
            plt.polar(angles, similar_values, 'o-', label='Similar Item')
            
            # Set feature labels
            plt.xticks(angles[:-1], common_features)
            
            plt.title('Feature Comparison')
            plt.legend()
            
        elif plot_type == 'parallel':
            # Create parallel coordinates plot
            plt.figure(figsize=(12, 6))
            
            for i, feature in enumerate(common_features):
                # Plot lines between features
                if i < len(common_features) - 1:
                    # Query line
                    plt.plot(
                        [i, i+1], 
                        [query_features[common_features[i]], query_features[common_features[i+1]]], 
                        'b-', alpha=0.7
                    )
                    # Similar item line
                    plt.plot(
                        [i, i+1], 
                        [similar_item[common_features[i]], similar_item[common_features[i+1]]], 
                        'r-', alpha=0.7
                    )
                
                # Plot points
                plt.plot(i, query_features[common_features[i]], 'bo', label='Query' if i == 0 else "")
                plt.plot(i, similar_item[common_features[i]], 'ro', label='Similar Item' if i == 0 else "")
            
            plt.xticks(range(len(common_features)), common_features, rotation=45)
            plt.title('Feature Comparison')
            plt.legend()
            
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Feature comparison plot saved to {output_path}")
            
        plt.close()
    
    def save(self, output_path: str):
        """
        Save the similarity model.
        
        Args:
            output_path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        logger.info(f"Saving similarity model to {output_path}")
        joblib.dump(self, output_path)
    
    @classmethod
    def load(cls, input_path: str):
        """
        Load a similarity model.
        
        Args:
            input_path: Path to the saved model
            
        Returns:
            Loaded SimilaritySearch object
        """
        logger.info(f"Loading similarity model from {input_path}")
        return joblib.load(input_path)


def create_similarity_model(X: np.ndarray, df: pd.DataFrame, 
                          feature_names: List[str],
                          metric: str = 'euclidean',
                          model_path: Optional[str] = None) -> SimilaritySearch:
    """
    Create and fit a similarity search model.
    
    Args:
        X: Feature matrix
        df: Original DataFrame
        feature_names: Names of features
        metric: Distance metric ('euclidean' or 'cosine')
        model_path: Path to save the model
        
    Returns:
        Fitted SimilaritySearch object
    """
    logger.info(f"Creating similarity model with {metric} metric")
    
    # Create model
    similarity_model = SimilaritySearch(feature_names, metric)
    
    # Fit model
    similarity_model.fit(X, df)
    
    # Save model if path provided
    if model_path:
        similarity_model.save(model_path)
    
    return similarity_model


if __name__ == "__main__":
    # Example usage
    # from preprocessing import preprocess_data
    # from features import engineer_features, get_feature_names
    # 
    # # Preprocess data
    # df = preprocess_data("../data/db_computers_2025_raw.csv")
    # 
    # # Engineer features
    # X_transformed, pipeline = engineer_features(df)
    # 
    # # Get feature names
    # numeric_features, categorical_features = identify_core_features(df)
    # feature_names = get_feature_names(pipeline, numeric_features, categorical_features)
    # 
    # # Create similarity model
    # similarity_model = create_similarity_model(
    #     X_transformed, 
    #     df,
    #     feature_names,
    #     metric='euclidean',
    #     model_path="../models/similarity_model.joblib"
    # )
    pass
