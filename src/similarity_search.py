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
    
    # Define features for similarity comparison
    features = ['cpu_rating', 'ram', 'storage', 'screen_size', 'has_dedicated_gpu',
                'gpu_rating', 'battery_life', 'is_touchscreen', 'weight']
    
    # Ensure all required features exist
    missing_features = [f for f in features if f not in computers_df.columns]
    if missing_features:
        # Create placeholder features with default values
        for feature in missing_features:
            computers_df[feature] = 0
    
    # Load or create feature scaler
    scaler_path = os.path.join(MODEL_DIR, "similarity_scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(computers_df[features])
        # Save for future use
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(scaler, scaler_path)
    
    # Scale database features
    X = scaler.transform(computers_df[features])
    
    # Prepare user config
    user_vector = prepare_config_for_similarity(config_dict, scaler)
    
    # Find nearest neighbors
    nn_model = NearestNeighbors(n_neighbors=min(top_n + 1, len(computers_df)), metric='euclidean')
    nn_model.fit(X)
    
    distances, indices = nn_model.kneighbors(user_vector)
    
    # Get similar computers (skip first result if it's identical to user config)
    similar_computers = computers_df.iloc[indices[0][1:top_n+1]]
    similarity_scores = 1 - (distances[0][1:top_n+1] / distances[0].max())
    
    # Add similarity score to results
    similar_computers = similar_computers.copy()
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
    st.header("🔍 Similar Configurations")
    
    st.markdown("""
    <div style="padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.1);">
    <h4>Find Alternative Configurations</h4>
    <p>Discover computers that are similar to your current configuration. 
    These alternatives may offer different price points or features while still meeting your needs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Find similar computers
    similar_computers = find_similar_computers(user_config, top_n=5)
    
    if similar_computers.empty:
        st.warning("No similar configurations found. Try adjusting your configuration.")
        return
    
    # Display the similar computers
    for idx, config in similar_computers.iterrows():
        similarity = config['similarity_score'] * 100
        price = config.get('price', "N/A")
        brand = config.get('brand', "Unknown")
        model = config.get('model', "Computer")
        
        # Create a visually appealing card for each similar computer
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Similarity score with a circular progress indicator
            st.markdown(f"""
            <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
            <div style="position: relative; width: 100px; height: 100px;">
              <svg width="100" height="100" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="45" fill="none" stroke="#e0e0e0" stroke-width="10"></circle>
                <circle cx="50" cy="50" r="45" fill="none" stroke="#4CAF50" stroke-width="10" 
                        stroke-dasharray="283" stroke-dashoffset="{283 - (283 * similarity / 100)}" 
                        transform="rotate(-90 50 50)"></circle>
              </svg>
              <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                        font-size: 20px; font-weight: bold;">
                {similarity:.0f}%
              </div>
            </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Computer details
            st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50; 
                     background-color: rgba(255,255,255,0.05); margin-bottom: 10px;">
                <h3>{brand} {model}</h3>
                <p><strong>Price:</strong> €{price if price != 'N/A' else 'N/A'}</p>
                <p>
                    <span style="margin-right: 15px;">🔄 CPU: {config.get('cpu_rating', 'N/A')}/10</span>
                    <span style="margin-right: 15px;">🧠 RAM: {config.get('ram', 'N/A')}GB</span>
                    <span style="margin-right: 15px;">💾 Storage: {config.get('storage', 'N/A')}GB</span>
                    <span>🖥️ Screen: {config.get('screen_size', 'N/A')}"</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Why this computer is similar
            explanations = calculate_similarity_explanation(user_config, config)
            if explanations:
                with st.expander("Why this is similar?"):
                    for explanation in explanations:
                        st.markdown(f"{explanation['icon']} **{explanation['feature']}:** {explanation['match']}")
    
    # Add a section for comparing user's configuration with the alternatives
    st.subheader("How Alternatives Compare")
    
    try:
        # Create a radar chart to compare key attributes
        if not similar_computers.empty:
            features_to_compare = ['cpu_rating', 'ram', 'storage', 'gpu_rating', 'battery_life']
            labels = ['CPU Performance', 'RAM', 'Storage', 'GPU Performance', 'Battery Life']
            
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
            
            # Extract top 3 similar computers
            top_similar = similar_computers.head(3)
            
            # Create radar chart data
            fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
            
            # Number of variables
            N = len(labels)
            
            # What will be the angle of each axis in the plot (divide the plot / number of variables)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Plot for user configuration
            user_values += user_values[:1]  # Close the loop
            ax.plot(angles, user_values, linewidth=2, linestyle='solid', label="Your Configuration")
            ax.fill(angles, user_values, alpha=0.1)
            
            # Plot for each similar computer
            colors = ['#FF9999', '#66B2FF', '#99FF99']
            for i, (_, config) in enumerate(top_similar.iterrows()):
                values = []
                for feature in features_to_compare:
                    values.append(config.get(feature, 0))
                
                # Normalize
                values = [min(1.0, values[i] / max_values[i]) for i in range(len(values))]
                values += values[:1]  # Close the loop
                
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Alternative {i+1}", 
                        color=colors[i % len(colors)])
                ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
            
            # Add labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            
            # Remove radial labels
            ax.set_yticklabels([])
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            plt.title('Configuration Comparison', size=15, pad=20)
            
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error generating comparison visualization: {e}")
        
    # Provide insights about the alternatives
    st.markdown("### Insights")
    st.markdown("""
    - **Price Range**: The alternatives offer price points that may be lower or higher based on different feature combinations.
    - **Performance Trade-offs**: Some alternatives may offer better performance in specific areas while compromising in others.
    - **Brand Variations**: Different manufacturers may offer similar specifications at varying price points due to brand premium.
    """)
    
    # Add a feedback section
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.button("👍 These alternatives are helpful", key="helpful_alternatives")
    with col2:
        st.button("👎 These alternatives aren't relevant", key="irrelevant_alternatives")
    with col3:
        st.button("🔄 Show more alternatives", key="more_alternatives")

def render_similarity_search_tab(user_config: Dict):
    """
    Render the Similarity Search tab in the Streamlit UI.
    
    Args:
        user_config: Dictionary with user's computer configuration
    """
    render_similar_computers(user_config)
