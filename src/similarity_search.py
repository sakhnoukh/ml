"""
Similarity Search Module for ML Marketplace

This module provides functionality for finding computer configurations that are
similar to a user's specified configuration, helping users discover alternatives
and make comparisons.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from typing import Dict, List, Any, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Define the absolute paths to important directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
DATA_DIR = os.path.join(ROOT_DIR, "data")

def prepare_config_for_similarity(config_dict: Dict, scaler=None) -> np.ndarray:
    """
    Prepare a user configuration for similarity search by converting
    it to a feature vector compatible with our models.
    
    Args:
        config_dict: Dictionary with user's computer configuration
        scaler: Optional pre-fit scaler for feature normalization
        
    Returns:
        Feature vector as numpy array
    """
    # Key features for similarity matching
    features = ['cpu_rating', 'ram', 'storage', 'screen_size', 'has_dedicated_gpu',
                'gpu_rating', 'battery_life', 'is_touchscreen', 'weight']
    
    # Create a dataframe with the user's configuration
    user_config = {}
    
    # Extract and transform features from the config dictionary
    user_config['cpu_rating'] = config_dict.get('cpu_rating', 5)
    user_config['ram'] = config_dict.get('ram', 8)
    user_config['storage'] = config_dict.get('storage', 512)
    user_config['screen_size'] = config_dict.get('screen_size', 15.6)
    user_config['has_dedicated_gpu'] = 1 if 'graphics_card' in config_dict and config_dict['graphics_card'] != 'Integrated Graphics' else 0
    user_config['gpu_rating'] = config_dict.get('gpu_rating', 0)
    user_config['battery_life'] = config_dict.get('battery_life', 8)
    user_config['is_touchscreen'] = 1 if config_dict.get('has_touchscreen', False) else 0
    user_config['weight'] = config_dict.get('weight', 2.0)
    
    # Convert to DataFrame for easier processing
    user_df = pd.DataFrame([user_config])
    
    # Scale features if a scaler is provided
    if scaler:
        user_vector = scaler.transform(user_df)
    else:
        user_vector = user_df.values
    
    return user_vector

def find_similar_computers(config_dict: Dict, top_n: int = 5) -> pd.DataFrame:
    """
    Find similar computer configurations based on the user's configuration.
    
    Args:
        config_dict: Dictionary with user's computer configuration
        top_n: Number of similar configurations to retrieve
        
    Returns:
        DataFrame with similar configurations
    """
    # Load training data
    computers_df_path = os.path.join(DATA_DIR, "processed_computers.csv")
    if not os.path.exists(computers_df_path):
        st.error("Computer database not found. Cannot find similar configurations.")
        return pd.DataFrame()
    
    computers_df = pd.read_csv(computers_df_path)
    
    # Handle brand matching specifically for Apple
    is_apple_cpu = config_dict.get('cpu_brand', '').lower() == 'apple'
    
    # If user selected an Apple CPU, prioritize MacBooks
    if is_apple_cpu:
        # Filter to only Apple products first
        apple_computers = computers_df[computers_df['brand'] == 'Apple']
        
        # If we have Apple products in our database
        if not apple_computers.empty:
            # We'll still do similarity matching, but only within Apple products
            top_apple_matches = find_similar_within_subset(config_dict, apple_computers, min(top_n, len(apple_computers)))
            
            # If we need more results to meet the requested top_n, get non-Apple alternatives
            if len(top_apple_matches) < top_n:
                non_apple_computers = computers_df[computers_df['brand'] != 'Apple']
                remaining_matches = find_similar_within_subset(
                    config_dict, 
                    non_apple_computers, 
                    top_n - len(top_apple_matches)
                )
                # Combine the results, with Apple matches first
                return pd.concat([top_apple_matches, remaining_matches])
            
            # We have enough Apple matches
            return top_apple_matches
    
    # For non-Apple or if no Apple products are found, use the standard approach
    return find_similar_within_subset(config_dict, computers_df, top_n)


def find_similar_within_subset(config_dict: Dict, computers_subset: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Find similar configurations within a specific subset of computers (e.g., only Apple products).
    
    Args:
        config_dict: Dictionary with user's computer configuration
        computers_subset: DataFrame subset to search within
        top_n: Number of similar configurations to retrieve
        
    Returns:
        DataFrame with similar configurations
    """
    if computers_subset.empty or top_n <= 0:
        return pd.DataFrame()
    
    # Define features for similarity comparison
    features = ['cpu_rating', 'ram', 'storage', 'screen_size', 'has_dedicated_gpu',
                'gpu_rating', 'battery_life', 'is_touchscreen', 'weight']
    
    # Ensure all required features exist
    missing_features = [f for f in features if f not in computers_subset.columns]
    if missing_features:
        # Create placeholder features with default values
        for feature in missing_features:
            computers_subset[feature] = 0
    
    # Create a scaler specifically for this subset
    scaler = StandardScaler()
    scaler.fit(computers_subset[features])
    
    # Scale database features
    X = scaler.transform(computers_subset[features])
    
    # Prepare user config
    user_vector = prepare_config_for_similarity(config_dict, scaler)
    
    # Find nearest neighbors
    n_neighbors = min(top_n + 1, len(computers_subset))
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn_model.fit(X)
    
    distances, indices = nn_model.kneighbors(user_vector)
    
    # Get similar computers (use all results as we're already working with a subset)
    max_idx = min(n_neighbors, len(indices[0]))
    result_indices = indices[0][:max_idx]
    result_distances = distances[0][:max_idx]
    
    # Normalize distances to similarity scores
    if len(result_distances) > 0 and result_distances.max() > 0:
        similarity_scores = 1 - (result_distances / result_distances.max())
    else:
        similarity_scores = np.ones(len(result_indices))
    
    # Add similarity score to results
    similar_computers = computers_subset.iloc[result_indices].copy()
    similar_computers['similarity_score'] = similarity_scores
    
    return similar_computers

def calculate_similarity_explanation(user_config: Dict, similar_config: pd.Series) -> Dict:
    """
    Calculate an explanation of why a computer is similar to the user's configuration.
    
    Args:
        user_config: Dictionary with user's computer configuration
        similar_config: Series representing a similar computer's configuration
        
    Returns:
        Dictionary with similarity explanations
    """
    explanations = []
    
    # CPU comparison
    user_cpu = user_config.get('cpu_rating', 5)
    similar_cpu = similar_config.get('cpu_rating', 5)
    if abs(user_cpu - similar_cpu) <= 1:
        explanations.append({
            "feature": "CPU",
            "match": "Similar CPU performance",
            "icon": "🔄"
        })
    
    # RAM comparison
    user_ram = user_config.get('ram', 8)
    similar_ram = similar_config.get('ram', 8)
    if abs(user_ram - similar_ram) <= 4:
        explanations.append({
            "feature": "RAM",
            "match": f"Similar memory: {similar_ram}GB vs {user_ram}GB",
            "icon": "🧠"
        })
    
    # GPU comparison
    user_has_gpu = 'graphics_card' in user_config and user_config['graphics_card'] != 'Integrated Graphics'
    similar_has_gpu = similar_config.get('has_dedicated_gpu', 0) == 1
    if user_has_gpu == similar_has_gpu:
        if user_has_gpu:
            explanations.append({
                "feature": "GPU",
                "match": "Both have dedicated graphics",
                "icon": "🎮"
            })
        else:
            explanations.append({
                "feature": "GPU",
                "match": "Both use integrated graphics",
                "icon": "🎮"
            })
    
    # Screen size comparison
    user_screen = user_config.get('screen_size', 15.6)
    similar_screen = similar_config.get('screen_size', 15.6)
    if abs(user_screen - similar_screen) <= 1:
        explanations.append({
            "feature": "Screen",
            "match": f"Similar display size: {similar_screen}\"",
            "icon": "🖥️"
        })
    
    # Storage comparison
    user_storage = user_config.get('storage', 512)
    similar_storage = similar_config.get('storage', 512)
    storage_ratio = max(user_storage, similar_storage) / min(user_storage, similar_storage)
    if storage_ratio < 2:  # Within 2x storage capacity
        explanations.append({
            "feature": "Storage",
            "match": f"Comparable storage capacity: {similar_storage}GB",
            "icon": "💾"
        })
    
    return explanations

def render_similar_computers(user_config: Dict):
    """
    Render the similar computers UI in the Streamlit app.
    
    Args:
        user_config: Dictionary with user's computer configuration
    """
    
    # Find similar computers
    similar_computers = find_similar_computers(user_config, top_n=5)
    
    if similar_computers.empty:
        st.warning("No similar configurations found. Try adjusting your configuration.")
        return
    
    # Display the similar computers using a more compact layout
    if not similar_computers.empty:
        # Use columns to display computers side by side
        cols = st.columns(3)
        
        for i, (idx, config) in enumerate(similar_computers.head(3).iterrows()):
            similarity = config['similarity_score'] * 100
            price = config.get('price', "N/A")
            brand = config.get('brand', "Unknown")
            model = config.get('model', "Computer")
            
            with cols[i % 3]:
                # Create a compact card for each similar computer
                st.markdown(f"""
                <div style="padding: 10px; border-radius: 8px; border-left: 4px solid #4CAF50; 
                         background-color: rgba(255,255,255,0.05); height: 100%;">
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <div style="flex-grow: 1;"><strong>{brand} {model}</strong></div>
                        <div style="background-color: #4CAF50; color: white; padding: 3px 8px; 
                                 border-radius: 12px; font-size: 0.8em; margin-left: 5px;">
                            {similarity:.0f}% match
                        </div>
                    </div>
                    <p style="margin: 5px 0;"><strong>Price:</strong> €{price if price != 'N/A' else 'N/A'}</p>
                    <p style="margin: 5px 0; font-size: 0.9em;">
                        🔄 CPU: {config.get('cpu_rating', 'N/A')}/10 • 
                        🧠 {config.get('ram', 'N/A')}GB • 
                        💾 {config.get('storage', 'N/A')}GB
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Why this computer is similar - condensed display
                explanations = calculate_similarity_explanation(user_config, config)
                if explanations:
                    with st.expander("Why similar?"):
                        for explanation in explanations:
                            st.markdown(f"{explanation['icon']} {explanation['match']}", help=f"Feature: {explanation['feature']}")
    
        # Show a "View more" button below the 3 cards
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.button("🔍 View more alternatives", key="view_more_computers")
    
    # Add a compact comparison section
    with st.expander("📊 Compare Specifications", expanded=False):
        st.markdown("See how your configuration compares with similar alternatives")
        
        try:
            # Create a radar chart to compare key attributes
            if not similar_computers.empty:
                features_to_compare = ['cpu_rating', 'ram', 'storage', 'gpu_rating', 'battery_life']
                labels = ['CPU', 'RAM', 'Storage', 'GPU', 'Battery']
                
                # Extract user config values
                user_values = []
                for feature in features_to_compare:
                    if feature == 'has_dedicated_gpu':
                        user_values.append(1 if user_config.get('graphics_card', 'Integrated Graphics') != 'Integrated Graphics' else 0)
                    else:
                        user_values.append(user_config.get(feature, 0))
                
                # Normalize the values for better visualization
                max_values = [10, 64, 2000, 10, 24]  # Maximum reasonable values for each feature
                user_values = [min(1.0, user_values[i] / max_values[i]) for i in range(len(user_values))]
                
                # Extract top 2 similar computers to avoid crowding
                top_similar = similar_computers.head(2)
                
                # Create radar chart data
                fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(polar=True))
                
                # Number of variables
                N = len(labels)
                
                # What will be the angle of each axis in the plot
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Close the loop
                
                # Plot for user configuration
                user_values += user_values[:1]  # Close the loop
                ax.plot(angles, user_values, linewidth=2, linestyle='solid', label="Your Config.")
                ax.fill(angles, user_values, alpha=0.1)
                
                # Plot for each similar computer
                colors = ['#FF9999', '#66B2FF']
                for i, (_, config) in enumerate(top_similar.iterrows()):
                    values = []
                    for feature in features_to_compare:
                        values.append(config.get(feature, 0))
                    
                    # Normalize
                    values = [min(1.0, values[i] / max_values[i]) for i in range(len(values))]
                    values += values[:1]  # Close the loop
                    
                    ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"{config['brand']} {config['model']}", 
                            color=colors[i % len(colors)])
                    ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
                
                # Add labels
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(labels)
                
                # Remove radial labels
                ax.set_yticklabels([])
                
                # Add legend with smaller font
                plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize='small')
                
                plt.title('Specification Comparison', size=12)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Explain how similarity is calculated
                st.info("**How similarity is calculated**: The k-Nearest Neighbors (k-NN) algorithm computes Euclidean distances between feature vectors after normalizing specifications by their typical ranges. CPU rating, RAM, storage, screen size, and GPU capabilities are weighted more heavily in the computation.")
                
        except Exception as e:
            st.error(f"Error generating comparison visualization: {e}")
    
    # Add a compact feedback section
    st.write("")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("👍 Helpful alternatives", key="helpful_alternatives")
    with col2:
        st.button("👎 Not relevant", key="irrelevant_alternatives")

def render_similarity_search_tab(user_config: Dict):
    """
    Render the Similarity Search tab in the Streamlit UI.
    
    Args:
        user_config: Dictionary with user's computer configuration
    """
    render_similar_computers(user_config)
