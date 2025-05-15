import streamlit as st
import pandas as pd
import numpy as np
import joblib

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
        base_price_score += 1.0 # Significant Apple premium

    # CPU Rating (1-10 scale)
    # More granular impact for CPU
    if specs['cpu_rating'] <= 3: # Low-end
        base_price_score += (specs['cpu_rating'] * 0.05) 
    elif specs['cpu_rating'] <= 7: # Mid-range
        base_price_score += 0.15 + ((specs['cpu_rating'] - 3) * 0.1) # 0.15 to 0.55
    else: # High-end
        base_price_score += 0.55 + ((specs['cpu_rating'] - 7) * 0.3) # 0.55 to 1.45 (max for rating 10)

    # RAM (GB) - Tiered impact
    if specs['ram'] <= 4:
        base_price_score += 0.05
    elif specs['ram'] == 8:
        base_price_score += 0.2
    elif specs['ram'] == 16:
        base_price_score += 0.4
    elif specs['ram'] == 32:
        base_price_score += 0.7
    elif specs['ram'] >= 64:
        base_price_score += 1.0

    # SSD Capacity (GB) - Tiered impact
    if specs['ssd_capacity'] <= 128:
        base_price_score += 0.05
    elif specs['ssd_capacity'] == 256:
        base_price_score += 0.2
    elif specs['ssd_capacity'] == 512:
        base_price_score += 0.4
    elif specs['ssd_capacity'] == 1024: # 1TB
        base_price_score += 0.7
    elif specs['ssd_capacity'] >= 2048: # 2TB+
        base_price_score += 1.0

    # Screen Size (inches) - Minor impact, larger slightly more
    if specs['screen_size'] >= 17.0:
        base_price_score += 0.2
    elif specs['screen_size'] >= 15.0:
        base_price_score += 0.1

    # Screen Resolution - Significant impact for higher resolutions
    if specs['screen_resolution'] == "QHD (2560x1440)":
        base_price_score += 0.3
    elif specs['screen_resolution'] == "4K (3840x2160)":
        base_price_score += 0.7 # Significant premium for 4K

    # Graphics Card - Major impact, especially for dedicated GPUs
    # Using a more detailed scoring for GPUs
    gpu_scores = {
        "Integrated Graphics": 0.0,
        "NVIDIA GeForce MX350": 0.3, # Entry
        "NVIDIA GeForce GTX 1650": 0.6, # Budget Gaming
        "NVIDIA GeForce RTX 3050": 0.9, # Mid-Laptop
        "NVIDIA GeForce RTX 4050": 1.1, # Newer Mid-Laptop
        "AMD Radeon RX 6600M": 1.0,    # Mid-Laptop AMD
        "NVIDIA GeForce RTX 3060": 1.5, # Upper Mid
        "NVIDIA GeForce RTX 4060": 1.8  # High-End Laptop
        # Desktop GPUs would be higher, but this is laptop focused
    }
    base_price_score += gpu_scores.get(specs['graphics_card'], 0.0)


    # Touchscreen - Minor premium
    if specs['has_touchscreen']:
        base_price_score += 0.15

    # Battery Life (hours) - Minor impact for very long battery
    if specs['battery_life'] >= 10:
        base_price_score += 0.1
    if specs['battery_life'] >= 15: # Premium for very long
        base_price_score += 0.1 # total 0.2

    # Ensure price score is within 0-5 range
    price_scale = max(0, min(base_price_score, 5.0))
    return round(price_scale, 1)

# Load the trained model and scaler (if you saved one)
# model = joblib.load('models/computer_price_model.joblib') # We are not using the ML model for now
# scaler = joblib.load('models/scaler.joblib') # Assuming you saved a scaler

st.set_page_config(layout="wide")
st.title("💻 ML Marketplace: Computer Price Predictor")

# --- Sidebar for About and general info ---
st.sidebar.header("About")
st.sidebar.info(
    "This app predicts computer prices based on their specifications. "
    "The prediction uses a rule-based system fine-tuned with market data, "
    "offering a more realistic price estimation than a pure ML model for this specific dataset."
)
# Input features will now be in the main column using expanders

# Define discrete options for RAM, SSD, and Graphics Cards
ram_options = [4, 8, 16, 32, 64]  # GB
ssd_options = [128, 256, 512, 1024, 2048]  # GB (1024=1TB, 2048=2TB)

# Enriched graphics card options: Display Name -> Internal Value
graphics_card_options_map = {
    "Integrated Graphics (Intel Iris Xe / AMD Radeon)": "Integrated Graphics",
    "NVIDIA GeForce MX350 (Entry-Level, 2GB GDDR5)": "NVIDIA GeForce MX350",
    "NVIDIA GeForce GTX 1650 (Budget Gaming, 4GB GDDR6)": "NVIDIA GeForce GTX 1650",
    "NVIDIA GeForce RTX 3050 (Mid-Range Laptop, 4GB GDDR6)": "NVIDIA GeForce RTX 3050", 
    "NVIDIA GeForce RTX 4050 (Mid-Range Laptop, 6GB GDDR6)": "NVIDIA GeForce RTX 4050",
    "AMD Radeon RX 6600M (Mid-Range Laptop, 8GB GDDR6)": "AMD Radeon RX 6600M",
    "NVIDIA GeForce RTX 3060 (Upper Mid-Range Laptop, 6GB GDDR6)": "NVIDIA GeForce RTX 3060",
    "NVIDIA GeForce RTX 4060 (High-End Laptop, 8GB GDDR6)": "NVIDIA GeForce RTX 4060",
    "AMD Radeon RX 6800M (High-End Laptop, 12GB GDDR6)": "AMD Radeon RX 6800M"
}
screen_resolution_options = {
    "HD (1366x768)": "HD (1366x768)",
    "Full HD (1920x1080)": "Full HD (1920x1080)",
    "QHD (2560x1440)": "QHD (2560x1440)", 
    "4K (3840x2160)": "4K (3840x2160)"
}
cpu_brand_options = ["Intel", "AMD", "Apple"]

# --- UI Layout --- #
col1, col2 = st.columns([0.6, 0.4]) # Adjusted column ratio

with col1:
    st.subheader("Configure Your Computer Hardware")
    
    with st.expander("⚙️ Processor & Memory", expanded=True):
        cpu_brand = st.selectbox("CPU Brand", options=cpu_brand_options, index=0, help="Select the CPU manufacturer.")
        cpu_rating = st.slider("CPU Performance Rating (1-10)", 1, 10, 5, help="A general rating of CPU performance. Higher is better. (e.g., Intel Core i3 ~ 3-4, i5 ~ 5-7, i7/i9 ~ 8-10, Apple M1/M2 ~ 7-9 depending on variant)")
        ram = st.select_slider("RAM (GB)", options=ram_options, value=16, help="Select the amount of Random Access Memory (RAM). 8GB is good for basic use, 16GB for general productivity and light gaming, 32GB+ for demanding tasks.")

    with st.expander("💾 Storage, Display & Graphics", expanded=True):
        ssd_capacity = st.select_slider("SSD Capacity (GB)", options=ssd_options, value=512, help="Select the Solid State Drive (SSD) storage capacity. 256GB is basic, 512GB is common, 1TB+ for large files/games.")
        screen_size = st.slider("Screen Size (inches)", 13.0, 17.3, 15.6, step=0.1, help="Select the screen diagonal size in inches.")
        
        # Use descriptive names for options, but pass the simpler internal name to the backend
        selected_screen_resolution_display = st.selectbox(
            "Screen Resolution", 
            options=list(screen_resolution_options.keys()), 
            index=1, 
            help="Select the screen resolution. Higher resolutions offer sharper images."
        )
        screen_resolution = screen_resolution_options[selected_screen_resolution_display]

        selected_graphics_card_display = st.selectbox(
            "Graphics Card", 
            options=list(graphics_card_options_map.keys()), 
            index=4, # Default to a mid-range card
            help="Select the graphics card. Integrated graphics are basic, dedicated cards (NVIDIA/AMD) are for gaming and professional graphics work."
        )
        graphics_card = graphics_card_options_map[selected_graphics_card_display] # Get the internal value
        
    with st.expander("🔋 Other Features", expanded=False):
        has_touchscreen = st.checkbox("Touchscreen", value=False, help="Does the computer have a touchscreen interface?")
        battery_life = st.slider("Battery Life (hours)", 1, 20, 8, help="Estimated battery life in hours under typical usage.")

with col2:
    st.subheader("Price Estimation")
    
    # User inputs dictionary for prediction
    if cpu_brand and graphics_card and screen_resolution: # Ensure essential inputs are selected
        user_inputs = {
            'cpu_brand': cpu_brand,
            'cpu_rating': cpu_rating,
            'ram': ram,
            'ssd_capacity': ssd_capacity,
            'screen_size': screen_size,
            'screen_resolution': screen_resolution, # Pass the internal value
            'graphics_card': graphics_card, # Pass the internal value
            'has_touchscreen': has_touchscreen,
            'battery_life': battery_life
        }

        # Get prediction from the rule-based system
        prediction_score = calculate_price_directly(user_inputs)
        
        # Map to price range in euros
        price_ranges = {
            0: "€300-€500",
            1: "€500-€700",
            2: "€700-€1000",
            3: "€1000-€1500",
            4: "€1500-€2000",
            5: "€2000+"
        }
        
        # Refined logic for pushing to higher price categories
        # Apple base configuration floor
        if cpu_brand == "Apple" and ram <= 8 and ssd_capacity <= 256 and prediction_score < 2.8:
            prediction_score = 2.8 # Ensures base Macs hit at least €1000-€1500 range after rounding

        # High-end GPUs like RTX 4060 should ALWAYS push price into €1500+ range
        if graphics_card in ["NVIDIA GeForce RTX 4060", "NVIDIA GeForce RTX 3060", "AMD Radeon RX 6800M"] and prediction_score < 4.0:
            prediction_score = 4.0  # Force to at least €1500-€2000 range
        
        # Large RAM combined with large SSD should also be high-end
        if ram >= 32 and ssd_capacity >= 1024 and prediction_score < 3.5:
            prediction_score = 3.5  # Force to at least €1000-€1500 (closer to €1500)
            
        # 4K display should ensure at least upper mid-range pricing
        if screen_resolution == "4K (3840x2160)" and prediction_score < 3.0:
            prediction_score = 3.0  # Force to at least €1000-€1500 range
        
        # Premium Combination Effect: If multiple high-end components are present, give a boost
        premium_component_count = 0
        if graphics_card in ["NVIDIA GeForce RTX 4060", "NVIDIA GeForce RTX 3060", "AMD Radeon RX 6800M"]: premium_component_count +=1
        if ram >= 32: premium_component_count +=1
        if ssd_capacity >= 1024: premium_component_count +=1 # 1TB or more
        if cpu_rating >= 8 : premium_component_count +=1
        if screen_resolution == "4K (3840x2160)": premium_component_count +=1
        
        if premium_component_count >= 4 and prediction_score < 4.75: # 4+ very high-end components
             prediction_score = min(prediction_score + 0.75, 5.0) 
        elif premium_component_count >= 3 and prediction_score < 4.25: # 3 high-end components
            prediction_score = min(prediction_score + 0.5, 5.0) # Boost, but cap at 5.0
        elif premium_component_count >=2 and prediction_score < 3.75: # 2 high-end components
             prediction_score = min(prediction_score + 0.25, 5.0)
            
        # Get the closest price range index
        price_index = round(prediction_score) # Round to nearest integer for indexing
        price_index = min(max(price_index, 0), 5) # Clamp between 0 and 5
        price_range_str = price_ranges[price_index]

        st.metric(label="Predicted Price Range", value=price_range_str)
        
        # Progress bar should reflect the raw score before rounding for index
        st.progress( (prediction_score / 5.0) ) # Max score is 5

        st.markdown(f"<sub>(Internal score: {prediction_score:.2f}/5.0)</sub>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("_Note: Prices are estimates and can vary based on brand, condition, and market fluctuations._")

    else:
        st.warning("Please configure the hardware specs to see a price estimation.")


st.markdown("---")
st.markdown("### Example Scenarios & Tips")
st.info(
    "💡 **Tip:** Select a high-end GPU like the 'NVIDIA GeForce RTX 4060', combine it with '32GB RAM', a '1TB SSD (1024GB)', a 'CPU Rating of 8+', and a '4K Screen' to see how premium configurations are priced. "
    "Conversely, 'Integrated Graphics' with '8GB RAM' and a '256GB SSD' will yield a budget-friendly estimate."
)
