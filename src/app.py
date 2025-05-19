import streamlit as st
import pandas as pd
import numpy as np
import joblib

import os
import sys

# Add the current directory to sys.path to ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the simplified action bar module
from action_bar_simple import get_action_bar_css, get_simple_bar_html

# Import the market segmentation module
from market_segmentation import render_market_segmentation_tab, identify_user_config_cluster

# Import the price breakdown module
from price_breakdown import display_price_breakdown

# Import the similarity search module
from similarity_search import render_similarity_search_tab

# Set page config
st.set_page_config(
    page_title="ML Marketplace: Computer Configuration",
    page_icon="💻",
    layout="wide"
)

# Add the CSS for the persistent action bar
st.markdown(get_action_bar_css(), unsafe_allow_html=True)


# Pre-defined configuration presets
PRESETS = {
    "Budget Friendly": {
        "cpu_brand": "Intel",
        "cpu_rating": 4,
        "ram": 8,
        "ssd_capacity": 256,
        "screen_size": 14.0,
        "screen_resolution": "FHD (1920 x 1080)",
        "graphics_card": "Integrated Graphics",
        "touchscreen": False,
        "battery_life": 6
    },
    "Balanced Performer": {
        "cpu_brand": "Intel",
        "cpu_rating": 7,
        "ram": 16,
        "ssd_capacity": 512,
        "screen_size": 15.6,
        "screen_resolution": "FHD (1920 x 1080)",
        "graphics_card": "NVIDIA GeForce GTX 1650",
        "touchscreen": False,
        "battery_life": 8
    },
    "Gaming Powerhouse": {
        "cpu_brand": "Intel",
        "cpu_rating": 9,
        "ram": 32,
        "ssd_capacity": 1024,
        "screen_size": 17.3,
        "screen_resolution": "QHD (2560 x 1440)",
        "graphics_card": "NVIDIA GeForce RTX 4060",
        "touchscreen": False,
        "battery_life": 4
    },
    "Ultra-Portable Creator": {
        "cpu_brand": "Apple",
        "cpu_rating": 8,
        "ram": 16,
        "ssd_capacity": 512,
        "screen_size": 13.3,
        "screen_resolution": "Retina (2560 x 1600)",
        "graphics_card": "Integrated Graphics",
        "touchscreen": False,  # Apple laptops don't have touchscreens
        "battery_life": 10
    }
}

# Function to apply a preset configuration
def apply_preset(preset_name):
    if preset_name in PRESETS:
        preset = PRESETS[preset_name]
        for key, value in preset.items():
            st.session_state[key] = value
        
        # Display a success message
        st.success(f"✅ Applied '{preset_name}' configuration!")

# Function to check compatibility between hardware components
def check_compatibility(specs):
    warnings = []
    
    # High-end GPU + Low battery life
    if specs['graphics_card'] in ["NVIDIA GeForce RTX 4060", "NVIDIA GeForce RTX 3060", "AMD Radeon RX 6800M"] and specs['battery_life'] < 5:
        warnings.append("⚠️ **Power Warning**: High-performance graphics cards significantly reduce battery life. Consider increasing battery capacity for mobility.")
    
    # 4K display + Low battery life
    if specs['screen_resolution'] == "4K (3840 x 2160)" and specs['battery_life'] < 6:
        warnings.append("⚠️ **Display Warning**: 4K displays consume more power. Your battery life will be shorter than estimated.")
    
    # Small screen + High resolution
    if specs['screen_size'] <= 14.0 and specs['screen_resolution'] in ["4K (3840 x 2160)", "QHD (2560 x 1440)"]:
        warnings.append("⚠️ **Usability Warning**: High resolution on a small screen may cause readability issues. Text and UI elements might appear too small.")
    
    # Low RAM + High-end GPU/CPU
    if specs['ram'] <= 8 and (specs['graphics_card'] in ["NVIDIA GeForce RTX 4060", "NVIDIA GeForce RTX 3060", "AMD Radeon RX 6800M"] or specs['cpu_rating'] >= 8):
        warnings.append("⚠️ **Performance Warning**: Low RAM (8GB or less) will bottleneck your high-performance CPU/GPU. Consider at least 16GB for balanced performance.")
    
    # Budget CPU + High-end GPU
    if specs['cpu_rating'] <= 5 and specs['graphics_card'] in ["NVIDIA GeForce RTX 4060", "NVIDIA GeForce RTX 3060", "AMD Radeon RX 6800M"]:
        warnings.append("⚠️ **Bottleneck Warning**: Your CPU may limit the performance of your high-end graphics card. Consider a higher-rated CPU.")
    
    # Apple CPU + non-Apple typical configurations
    if specs['cpu_brand'] == "Apple":
        if specs['ram'] > 64:
            warnings.append("⚠️ **Configuration Warning**: Apple computers typically don't offer more than 64GB RAM in most consumer models.")
        if specs['graphics_card'] not in ["Integrated Graphics"]:
            warnings.append("⚠️ **Configuration Warning**: Apple computers typically use integrated graphics or specific AMD models, not NVIDIA GPUs.")
        if specs['has_touchscreen']:
            warnings.append("⚠️ **Configuration Warning**: Apple laptops (MacBooks) don't have touchscreen functionality.")

    
    # Touchscreen + Large Screen
    if specs['has_touchscreen'] and specs['screen_size'] >= 17.0:
        warnings.append("⚠️ **Ergonomic Warning**: Touchscreens on larger displays (17+ inches) can be awkward to use due to arm fatigue.")
    
    return warnings

# Function to calculate price based on a more detailed rule-based system
def calculate_price_directly(specs):
    # Base price and component scores
    base_price_score = 0.0

    # CPU Brand Impact
    if specs['cpu_brand'] == "Intel":
        base_price_score += 0.2
    elif specs['cpu_brand'] == "AMD":
        base_price_score += 0.1
    elif specs['cpu_brand'] == "Apple":
        base_price_score += 0.8 # Apple premium (reduced from 1.0)

    # CPU Rating (1-10 scale)
    # More granular impact for CPU
    if specs['cpu_rating'] <= 3: # Low-end
        base_price_score += (specs['cpu_rating'] * 0.05) 
    elif specs['cpu_rating'] <= 7: # Mid-range
        base_price_score += 0.15 + ((specs['cpu_rating'] - 3) * 0.1) # 0.15 to 0.55
    else: # High-end (reduced impact for 8-10 ratings)
        base_price_score += 0.55 + ((specs['cpu_rating'] - 7) * 0.2) # 0.55 to 1.15 (max for rating 10)

    # RAM (GB) - Tiered impact (reduced impact for high-end RAM)
    if specs['ram'] <= 4:
        base_price_score += 0.05
    elif specs['ram'] == 8:
        base_price_score += 0.2
    elif specs['ram'] == 16:
        base_price_score += 0.35 # Reduced from 0.4
    elif specs['ram'] == 32:
        base_price_score += 0.5  # Reduced from 0.7
    elif specs['ram'] >= 64:
        base_price_score += 0.7  # Reduced from 1.0

    # SSD Capacity (GB) - Tiered impact (reduced impact for high capacity)
    if specs['ssd_capacity'] <= 128:
        base_price_score += 0.05
    elif specs['ssd_capacity'] == 256:
        base_price_score += 0.15 # Reduced from 0.2
    elif specs['ssd_capacity'] == 512:
        base_price_score += 0.3  # Reduced from 0.4
    elif specs['ssd_capacity'] == 1024: # 1TB
        base_price_score += 0.5  # Reduced from 0.7
    elif specs['ssd_capacity'] >= 2048: # 2TB+
        base_price_score += 0.7  # Reduced from 1.0

    # Screen Size (inches) - Minor impact, larger slightly more
    if specs['screen_size'] >= 17.0:
        base_price_score += 0.15 # Reduced from 0.2
    elif specs['screen_size'] >= 15.0:
        base_price_score += 0.1

    # Screen Resolution - More realistic impact for higher resolutions
    if specs['screen_resolution'] == "QHD (2560 x 1440)":
        base_price_score += 0.25 # Reduced from 0.3
    elif specs['screen_resolution'] == "4K (3840 x 2160)":
        base_price_score += 0.5  # Reduced from 0.7
    elif specs['screen_resolution'] == "Retina (2560 x 1600)":
        base_price_score += 0.3  # Reduced from 0.4

    # Graphics Card - More realistic pricing for dedicated GPUs
    # Using a more detailed scoring for GPUs with reduced premium
    gpu_scores = {
        "Integrated Graphics": 0.0,
        "NVIDIA GeForce MX450": 0.25,      # Entry (reduced from 0.3)
        "NVIDIA GeForce GTX 1650": 0.45,    # Budget Gaming (reduced from 0.6)
        "NVIDIA GeForce RTX 3050": 0.7,     # Mid-Laptop (reduced from 0.9)
        "NVIDIA GeForce RTX 4050": 0.8,     # Newer Mid-Laptop (reduced from 1.1)
        "AMD Radeon RX 6600M": 0.75,        # Mid-Laptop AMD (reduced from 1.0)
        "NVIDIA GeForce RTX 3060": 1.0,     # Upper Mid (reduced from 1.5)
        "NVIDIA GeForce RTX 4060": 1.2,     # High-End Laptop (reduced from 1.8)
        "AMD Radeon RX 6800M": 1.1          # High-End AMD
    }
    base_price_score += gpu_scores.get(specs['graphics_card'], 0.0)

    # Touchscreen - Minor premium
    if specs['has_touchscreen']:
        base_price_score += 0.2 # Reduced from 0.3
    
    # Longer battery life generally indicates more/better battery cells
    if specs['battery_life'] >= 10:
        base_price_score += 0.2  # Reduced from 0.3
    elif specs['battery_life'] >= 7:
        base_price_score += 0.1  # Reduced from 0.15
    
    # Apply GPU score if available for the selected card
    if specs['graphics_card'] in gpu_scores:
        base_price_score += gpu_scores[specs['graphics_card']]
    else: # Default to integrated if not found
        base_price_score += 0
    
    # Final scaling for the 0-6 scale (with more granularity at high end)
    return base_price_score


def main():
    # Sidebar
    st.sidebar.title("ML Marketplace")
    st.sidebar.image("https://img.icons8.com/fluency/96/laptop-settings.png", width=80)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### About")
    st.sidebar.info("This app predicts computer prices based on hardware specifications using machine learning.")
    st.sidebar.markdown("---")
    
    # Display title and introduction
    st.title("ML Marketplace: Computer Configuration")
    st.markdown(
        """Create your perfect computer configuration and get an estimated price range. 
        Use the smart presets for quick setups or customize each component."""
    )
    
    # Create tabs for different app sections
    tab1, tab2 = st.tabs(["💻 Configure Computer", "💹 Market Segmentation"])
    
    # Main content
    with tab1:
        st.title("💻 Computer Configuration")
        st.markdown("Configure your computer specifications and get a predicted price range.")
        
        # Initialize session state for all input widgets if not already set
        if "cpu_brand" not in st.session_state:
            st.session_state.cpu_brand = "Intel"
        if "cpu_rating" not in st.session_state:
            st.session_state.cpu_rating = 5
        if "ram" not in st.session_state:
            st.session_state.ram = 8
        if "ssd_capacity" not in st.session_state:
            st.session_state.ssd_capacity = 256
        if "screen_size" not in st.session_state:
            st.session_state.screen_size = 15.6
        if "screen_resolution" not in st.session_state:
            st.session_state.screen_resolution = "FHD (1920 x 1080)"
        if "graphics_card" not in st.session_state:
            st.session_state.graphics_card = "Integrated Graphics"
        if "touchscreen" not in st.session_state:
            st.session_state.touchscreen = False
        if "battery_life" not in st.session_state:
            st.session_state.battery_life = 6
        
        # Smart Presets section
        st.markdown("### 🚀 Quick Configuration Profiles")
        st.markdown("Select a preset configuration based on your needs:")
        
        preset_cols = st.columns(4)
        with preset_cols[0]:
            if st.button("💰 Budget Friendly", use_container_width=True):
                apply_preset("Budget Friendly")
        
        with preset_cols[1]:
            if st.button("⚖️ Balanced Performer", use_container_width=True):
                apply_preset("Balanced Performer")
        
        with preset_cols[2]:
            if st.button("🎮 Gaming Powerhouse", use_container_width=True):
                apply_preset("Gaming Powerhouse")
        
        with preset_cols[3]:
            if st.button("💼 Ultra-Portable Creator", use_container_width=True):
                apply_preset("Ultra-Portable Creator")
        
        st.markdown("---")
        
        # Main layout with two columns
        col1, col2 = st.columns([0.6, 0.4])
        
        # Hardware Configuration - Left Column
        with col1:
            # Processor & Memory Section
            with st.expander("⚙️ Processor & Memory", expanded=True):
                cpu_brand = st.selectbox(
                    "CPU Brand",
                    options=["Intel", "AMD", "Apple"],
                    help="Select the brand of the processor.",
                    key="cpu_brand"
                )
                
                cpu_rating = st.slider(
                    "CPU Performance Rating (1-10)",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.cpu_rating,
                    help="Indicate the performance tier of the CPU. Examples: Intel Core i3 ~ 3-4, Intel Core i5 ~ 5-7, Intel Core i7/i9 or Apple M1/M2 ~ 8-10",
                    key="cpu_rating"
                )
                
                ram = st.selectbox(
                    "RAM (GB)",
                    options=[4, 8, 16, 32, 64],
                    help="Select the amount of RAM in gigabytes.",
                    index=ram_options.index(st.session_state.ram) if st.session_state.ram in ram_options else 1,
                    key="ram"
                )
            
            # Storage, Display & Graphics Section
            with st.expander("💾 Storage, Display & Graphics", expanded=True):
                ssd_capacity = st.selectbox(
                    "SSD Capacity (GB)",
                    options=[128, 256, 512, 1024, 2048],
                    help="Select the capacity of the SSD storage in gigabytes.",
                    index=ssd_options.index(st.session_state.ssd_capacity) if st.session_state.ssd_capacity in ssd_options else 1,
                    key="ssd_capacity"
                )
                
                screen_size = st.selectbox(
                    "Screen Size (inches)",
                    options=[11.6, 13.3, 14.0, 15.6, 16.0, 17.3],
                    help="Select the display size in inches.",
                    index=2 if st.session_state.screen_size == 14.0 else 3,  # Default to 15.6"
                    key="screen_size"
                )
                
                # Screen resolution options
                screen_resolution_options = {
                    "HD (1366 x 768)": "HD (1366 x 768)",
                    "FHD (1920 x 1080)": "FHD (1920 x 1080)",
                    "QHD (2560 x 1440)": "QHD (2560 x 1440)",
                    "4K (3840 x 2160)": "4K (3840 x 2160)",
                    "Retina (2560 x 1600)": "Retina (2560 x 1600)"
                }
                
                screen_resolution = st.selectbox(
                    "Screen Resolution",
                    options=list(screen_resolution_options.keys()),
                    help="Select the screen resolution.",
                    index=list(screen_resolution_options.keys()).index(st.session_state.screen_resolution) if st.session_state.screen_resolution in screen_resolution_options.keys() else 1,
                    key="screen_resolution"
                )
                
                # Graphics card options with detailed descriptions but simple internal values
                graphics_card_options = {
                    "Integrated Graphics": "Integrated Graphics (Intel UHD/Iris, AMD Radeon, Apple Graphics)",
                    "NVIDIA GeForce MX450": "NVIDIA GeForce MX450 (Entry-Level, 2GB GDDR6)",
                    "NVIDIA GeForce GTX 1650": "NVIDIA GeForce GTX 1650 (Mid-Range, 4GB GDDR6)",
                    "NVIDIA GeForce RTX 3050": "NVIDIA GeForce RTX 3050 (Mid-High Range, 4GB GDDR6)",
                    "NVIDIA GeForce RTX 3060": "NVIDIA GeForce RTX 3060 (High-End, 6GB GDDR6)",
                    "NVIDIA GeForce RTX 4060": "NVIDIA GeForce RTX 4060 (High-End Laptop, 8GB GDDR6)",
                    "AMD Radeon RX 6600M": "AMD Radeon RX 6600M (Mid-Range, 8GB GDDR6)",
                    "AMD Radeon RX 6800M": "AMD Radeon RX 6800M (High-End, 12GB GDDR6)"
                }
                
                graphics_card = st.selectbox(
                    "Graphics Card",
                    options=list(graphics_card_options.keys()),
                    format_func=lambda x: graphics_card_options[x],
                    help="Select the graphics card model.",
                    index=list(graphics_card_options.keys()).index(st.session_state.graphics_card) if st.session_state.graphics_card in graphics_card_options.keys() else 0,
                    key="graphics_card"
                )
            
            # Other Features Section
            with st.expander("🔋 Other Features", expanded=True):
                touchscreen = st.checkbox(
                    "Touchscreen", 
                    help="Check if the laptop has a touchscreen display.",
                    value=st.session_state.touchscreen,
                    key="touchscreen"
                )
                
                battery_life = st.slider(
                    "Battery Life (hours)",
                    min_value=2,
                    max_value=24,
                    value=st.session_state.battery_life,
                    help="Estimated battery life in hours.",
                    key="battery_life"
                )            
        
            # Add the similarity search section after all the configuration expanders
            st.markdown("---")
            st.subheader("🔍 Similar Computers to Consider")
            
            # Display a brief explanation of the similarity search functionality
            st.markdown(
                """These are comparable commercial computer models that match your configuration. 
                Similarity is computed using the k-Nearest Neighbors algorithm based on hardware specifications."""
            )
            
            # Create a user_config dictionary based on the current configuration
            user_config = {
                'cpu_brand': cpu_brand,
                'cpu_rating': cpu_rating,
                'ram': ram,
                'storage': ssd_capacity,
                'screen_size': screen_size,
                'screen_resolution': screen_resolution,
                'graphics_card': graphics_card,
                'has_touchscreen': touchscreen,
                'weight': 2.0,  # Default weight in kg
                'battery_life': battery_life
            }
            
            # Render the similarity search section with the current configuration
            from similarity_search import render_similar_computers
            render_similar_computers(user_config)
        
        # Right Column - Price and Prediction
        with col2:
            st.subheader("Price Estimation")
            
            # User inputs dictionary for prediction
            user_inputs = {
                'cpu_brand': cpu_brand,
                'cpu_rating': cpu_rating,
                'ram': ram,
                'ssd_capacity': ssd_capacity,
                'screen_size': screen_size,
                'screen_resolution': screen_resolution,
                'graphics_card': graphics_card,
                'has_touchscreen': touchscreen,
                'battery_life': battery_life
            }

            # Get prediction from the rule-based system
            prediction_score = calculate_price_directly(user_inputs)
            
            # Map to price range in euros with more granularity at the high end
            price_ranges = {
                0: "€300-€500",
                1: "€500-€700",
                2: "€700-€1000",
                3: "€1000-€1500",
                4: "€1500-€2000",
                5: "€2000-€2500",
                6: "€2500+"
            }
            
            # Refined logic for pushing to higher price categories (reduced premium effect)
            # Apple base configuration floor (MacBook Air level)
            if cpu_brand == "Apple" and ram <= 8 and ssd_capacity <= 256 and prediction_score < 2.5:
                prediction_score = 2.5 # Base Macs start around €900-€1200

            # High-end GPUs pricing adjustment (more realistic ceiling)
            if graphics_card == "NVIDIA GeForce RTX 4060" and prediction_score < 3.5:
                prediction_score = 3.5  # Gaming laptops with RTX 4060 start ~€1500
            elif graphics_card in ["NVIDIA GeForce RTX 3060", "AMD Radeon RX 6800M"] and prediction_score < 3.0:
                prediction_score = 3.0  # Slightly lower tier
        
            # Large RAM combined with large SSD adjusted pricing
            if ram >= 32 and ssd_capacity >= 1024 and prediction_score < 3.0:
                prediction_score = 3.0  # Premium but not excessive
            
            # 4K display adjustment
            if screen_resolution == "4K (3840 x 2160)" and prediction_score < 2.8:
                prediction_score = 2.8  # 4K screens add premium but not extreme
        
            # Premium Combination Effect: More realistic premium effect
            premium_component_count = 0
            if graphics_card in ["NVIDIA GeForce RTX 4060", "NVIDIA GeForce RTX 3060", "AMD Radeon RX 6800M"]: 
                premium_component_count += 1
            if ram >= 32: 
                premium_component_count += 1
            if ssd_capacity >= 1024: 
                premium_component_count += 1 # 1TB or more
            if cpu_rating >= 8: 
                premium_component_count += 1
            if screen_resolution in ["4K (3840 x 2160)", "QHD (2560 x 1440)"]: 
                premium_component_count += 1
            
            # Gaming powerhouse configuration - target €1800-€2400 range (score ~4.0-5.0)
            # High-end laptop but not boutique premium pricing
            if premium_component_count >= 4: # 4+ high-end components
                # If it includes RTX 4060 + high CPU, it's a gaming powerhouse
                if graphics_card == "NVIDIA GeForce RTX 4060" and cpu_rating >= 8:
                    # Target score of ~4.5 (€2000-€2500 range)
                    if prediction_score < 4.0:
                        prediction_score = 4.0
                    elif prediction_score > 5.0: # Cap at reasonable price
                        prediction_score = 5.0
                else: # Other premium configs
                    prediction_score = min(prediction_score + 0.4, 5.0)
            elif premium_component_count >= 3: # 3 high-end components
                prediction_score = min(prediction_score + 0.3, 4.5) 
            elif premium_component_count >= 2: # 2 high-end components
                prediction_score = min(prediction_score + 0.2, 4.0)
            
            # Get the closest price range index
            price_index = round(prediction_score) # Round to nearest integer for indexing
            price_index = min(max(price_index, 0), 6) # Clamp between 0 and 6
            price_range_str = price_ranges[price_index]

            st.metric(label="Predicted Price Range", value=price_range_str)
            
            # Progress bar should reflect the raw score before rounding for index
            st.progress((prediction_score / 6.0)) # Max score is 6

            st.markdown(f"<sub>(Internal score: {prediction_score:.2f}/6.0)</sub>", unsafe_allow_html=True)
            
            # Add compatibility warnings
            st.markdown("---")
            st.subheader("Compatibility Check")
            
            # Check for compatibility issues
            compatibility_warnings = check_compatibility(user_inputs)
            
            if not compatibility_warnings:
                st.success("✅ No compatibility issues detected with your configuration.")
            else:
                for warning in compatibility_warnings:
                    st.warning(warning)
            
            # Add Price Breakdown section
            st.markdown("---")
            
            # Display price breakdown visualization
            display_price_breakdown(user_inputs, price_range_str)
            
            st.markdown("---")
            st.markdown("_Note: Prices are estimates and can vary based on brand, condition, and market fluctuations._")
            
            # Generate the configuration summary for the action bar
            config_summary = f"{cpu_brand} (Rating: {cpu_rating}) • {ram}GB RAM • {ssd_capacity}GB SSD"
            
            # Identify which market segment this configuration belongs to
            cluster_id, segment_name = identify_user_config_cluster({
                'cpu_brand': cpu_brand,
                'cpu_rating': cpu_rating,
                'ram': ram,
                'ssd_capacity': ssd_capacity,
                'screen_size': screen_size,
                'has_touchscreen': touchscreen,
                'battery_life': battery_life
            })
            
            # Include segment in summary if available
            if segment_name != "Unknown":
                config_summary += f" • {segment_name}"
            
            # Log this configuration to a feedback file (requirement from checklist)
            log_user_interaction({
                'timestamp': pd.Timestamp.now().isoformat(),
                'action': 'price_prediction',
                'cpu_brand': cpu_brand,
                'cpu_rating': cpu_rating,
                'ram': ram, 
                'ssd_capacity': ssd_capacity,
                'screen_size': screen_size,
                'screen_resolution': screen_resolution,
                'graphics_card': graphics_card,
                'touchscreen': touchscreen,
                'battery_life': battery_life,
                'estimated_price': price_range_str,
                'market_segment': segment_name
            })
            
            # Create and display the simple action bar with just a summary and price
            action_bar_html = get_simple_bar_html(config_summary, price_range_str)
            st.markdown(action_bar_html, unsafe_allow_html=True)
    
    # Market Segmentation Tab
    with tab2:
        render_market_segmentation_tab()

# Define RAM and SSD options (to be used in the main function)
ram_options = [4, 8, 16, 32, 64]
ssd_options = [128, 256, 512, 1024, 2048]


def log_user_interaction(interaction_data):
    """
    Log user interactions to a flat file for feedback analysis.
    This meets the flat-file feedback store requirement in the assignment.
    
    Args:
        interaction_data: Dictionary containing interaction details
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create or append to the feedback log file
    log_file = os.path.join(logs_dir, "user_interactions.csv")
    
    # Check if file exists to write header only once
    file_exists = os.path.isfile(log_file)
    
    # Convert interaction_data to DataFrame for easy CSV handling
    df = pd.DataFrame([interaction_data])
    
    # Write to CSV
    df.to_csv(log_file, mode='a', header=not file_exists, index=False)


# Run the main function
if __name__ == "__main__":
    main()
