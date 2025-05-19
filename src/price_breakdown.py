"""
Price Breakdown Module for ML Marketplace

This module provides functionality for breaking down and visualizing how different
hardware components contribute to the predicted price of a computer configuration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple
import datetime

def calculate_feature_contributions(config_dict):
    """
    Calculate how much each feature contributes to the final price.
    
    Args:
        config_dict: Dictionary with user's computer configuration
        
    Returns:
        DataFrame with feature contributions and total cost
    """
    contributions = []
    total_cost = 0
    
    # Base price - every computer has a minimum cost for case, assembly, etc.
    base_cost = 250  # Base cost in euros
    contributions.append({
        'feature': 'Base System',
        'contribution': base_cost,
        'percentage': 0,  # Will be calculated after all contributions are added
        'icon': '🖥️'
    })
    total_cost += base_cost
    
    # CPU Brand
    cpu_brand = config_dict.get('cpu_brand', '')
    cpu_brand_value = 0
    if cpu_brand == "Intel":
        cpu_brand_value = 80
    elif cpu_brand == "AMD":
        cpu_brand_value = 60
    elif cpu_brand == "Apple":
        cpu_brand_value = 300
    
    contributions.append({
        'feature': 'CPU Brand',
        'contribution': cpu_brand_value,
        'percentage': 0,
        'icon': '🔄'
    })
    total_cost += cpu_brand_value
    
    # CPU Performance Rating
    cpu_rating = config_dict.get('cpu_rating', 5)
    # Progressive scaling - higher ratings cost exponentially more
    cpu_rating_value = 0
    if cpu_rating <= 3:
        cpu_rating_value = cpu_rating * 30
    elif cpu_rating <= 7:
        cpu_rating_value = 90 + (cpu_rating - 3) * 60
    else:
        cpu_rating_value = 330 + (cpu_rating - 7) * 90
    
    contributions.append({
        'feature': 'CPU Performance',
        'contribution': cpu_rating_value,
        'percentage': 0,
        'icon': '⚡'
    })
    total_cost += cpu_rating_value
    
    # RAM
    ram = config_dict.get('ram', 8)
    # Calculate based on approximate market cost of RAM (€8/GB for first 16GB, then €6/GB)
    ram_value = 0
    if ram <= 16:
        ram_value = ram * 8
    else:
        ram_value = 16 * 8 + (ram - 16) * 6
        
    contributions.append({
        'feature': 'RAM',
        'contribution': ram_value,
        'percentage': 0,
        'icon': '🧠'
    })
    total_cost += ram_value
    
    # Storage
    ssd_capacity = config_dict.get('ssd_capacity', 512)
    # Calculate based on SSD pricing (€0.15/GB for first 1TB, then €0.10/GB)
    storage_value = 0
    if ssd_capacity <= 1024:
        storage_value = ssd_capacity * 0.15
    else:
        storage_value = 1024 * 0.15 + (ssd_capacity - 1024) * 0.10
        
    contributions.append({
        'feature': 'Storage',
        'contribution': storage_value,
        'percentage': 0,
        'icon': '💾'
    })
    total_cost += storage_value
    
    # Screen Size
    screen_size = config_dict.get('screen_size', 15.6)
    # Calculate screen cost based on size
    screen_size_value = 0
    if screen_size <= 14:
        screen_size_value = 80
    elif screen_size <= 15.6:
        screen_size_value = 100
    else:
        screen_size_value = 120 + (screen_size - 15.6) * 30
        
    contributions.append({
        'feature': 'Screen Size',
        'contribution': screen_size_value,
        'percentage': 0,
        'icon': '📏'
    })
    total_cost += screen_size_value
    
    # Screen Resolution
    screen_resolution = config_dict.get('screen_resolution', 'Full HD (1920 x 1080)')
    # Set values based on resolution
    resolution_value = 0
    if screen_resolution == "HD (1366 x 768)":
        resolution_value = 0  # No premium for HD
    elif screen_resolution == "Full HD (1920 x 1080)":
        resolution_value = 50
    elif screen_resolution == "QHD (2560 x 1440)":
        resolution_value = 150
    elif screen_resolution == "4K (3840 x 2160)":
        resolution_value = 300
    elif screen_resolution == "Retina (2560 x 1600)":
        resolution_value = 200
        
    contributions.append({
        'feature': 'Screen Resolution',
        'contribution': resolution_value,
        'percentage': 0,
        'icon': '🔍'
    })
    total_cost += resolution_value
    
    # Graphics Card
    graphics_card = config_dict.get('graphics_card', 'Integrated Graphics')
    # Set values based on GPU
    gpu_value = 0
    if graphics_card == "Integrated Graphics":
        gpu_value = 0
    elif graphics_card == "NVIDIA GeForce MX450":
        gpu_value = 120
    elif graphics_card == "NVIDIA GeForce GTX 1650":
        gpu_value = 200
    elif graphics_card == "NVIDIA GeForce RTX 3050":
        gpu_value = 300
    elif graphics_card == "NVIDIA GeForce RTX 4050":
        gpu_value = 400
    elif graphics_card == "AMD Radeon RX 6600M":
        gpu_value = 350
    elif graphics_card == "NVIDIA GeForce RTX 3060":
        gpu_value = 500
    elif graphics_card == "NVIDIA GeForce RTX 4060":
        gpu_value = 650
    elif graphics_card == "AMD Radeon RX 6800M":
        gpu_value = 550
        
    contributions.append({
        'feature': 'Graphics Card',
        'contribution': gpu_value,
        'percentage': 0,
        'icon': '🎮'
    })
    total_cost += gpu_value
    
    # Touchscreen
    has_touchscreen = config_dict.get('has_touchscreen', False)
    touchscreen_value = 100 if has_touchscreen else 0
    
    if has_touchscreen:
        contributions.append({
            'feature': 'Touchscreen',
            'contribution': touchscreen_value,
            'percentage': 0,
            'icon': '👆'
        })
        total_cost += touchscreen_value
    
    # Battery Life
    battery_life = config_dict.get('battery_life', 6)
    battery_value = max(0, (battery_life - 4) * 30)  # €30 per hour above 4 hours
    
    contributions.append({
        'feature': 'Battery',
        'contribution': battery_value,
        'percentage': 0,
        'icon': '🔋'
    })
    total_cost += battery_value
    
    # Calculate percentages
    for item in contributions:
        item['percentage'] = round((item['contribution'] / total_cost) * 100, 1)
    
    # Sort by contribution value (descending)
    contributions = sorted(contributions, key=lambda x: x['contribution'], reverse=True)
    
    # Create DataFrame
    df = pd.DataFrame(contributions)
    df['contribution'] = df['contribution'].round().astype(int)
    
    return df, total_cost

def log_user_feedback(feedback_type, config_dict):
    """
    Log user feedback on the price breakdown.
    
    Args:
        feedback_type: String indicating positive or negative feedback
        config_dict: Dictionary with user's computer configuration
    """
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "user_interactions.csv")
    
    # Create log file with headers if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("timestamp,feedback_type,cpu_brand,cpu_rating,ram,ssd_capacity,graphics_card\n")
    
    # Log the interaction
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cpu_brand = config_dict.get('cpu_brand', 'Unknown')
    cpu_rating = config_dict.get('cpu_rating', 0)
    ram = config_dict.get('ram', 0)
    ssd_capacity = config_dict.get('ssd_capacity', 0)
    graphics_card = config_dict.get('graphics_card', 'Unknown')
    
    log_entry = f"{timestamp},{feedback_type},{cpu_brand},{cpu_rating},{ram},{ssd_capacity},{graphics_card}\n"
    
    with open(log_file, 'a') as f:
        f.write(log_entry)

def display_price_breakdown(config_dict, predicted_price_range):
    """
    Display a detailed breakdown of the price components.
    
    Args:
        config_dict: Dictionary with user's computer configuration
        predicted_price_range: String containing the predicted price range
    """
    st.header("💰 Price Breakdown Analysis")
    
    st.write("""
    Understanding how each component impacts the final price can help you optimize your configuration.
    Below is a breakdown of estimated cost contributions from each component.
    """)
    
    # Calculate component contributions
    contributions_df, total_cost = calculate_feature_contributions(config_dict)
    
    # Extract the predicted price range
    min_price, max_price = extract_price_range(predicted_price_range)
    average_predicted = (min_price + max_price) / 2
    
    # Calculate any markup/margin
    model_markup = average_predicted - total_cost
    markup_percentage = (model_markup / total_cost) * 100 if total_cost > 0 else 0
    
    # Use a more spacious vertical layout with dedicated sections
    
    # Section 1: Price Summary Card (at the top for immediate visibility)
    st.subheader("Price Summary")
    
    # Add a stylized box for the price summary
    st.markdown(f"""
    <div style="background-color: rgba(255, 255, 255, 0.1); 
                padding: 20px; 
                border-radius: 10px; 
                border-left: 5px solid #4CAF50;
                margin-bottom: 30px;">
        <h3 style="margin-top: 0;">Estimated Total: €{total_cost:.0f}</h3>
        <p>Predicted range: {predicted_price_range}</p>
        <p>Model markup: €{model_markup:.0f} ({markup_percentage:.1f}%)</p>
        <small>*Actual prices may vary based on market conditions, brand premiums, and regional pricing.</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Section 2: Visual breakdown of component costs
    st.subheader("Component Cost Distribution")
    
    # Use a larger figure size for better readability
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar chart with improved styling
    colors = ['#ff9999' if x > 15 else '#9999ff' for x in contributions_df['percentage']]
    bars = ax.barh(contributions_df['feature'], contributions_df['contribution'], color=colors, height=0.6)
    
    # Add percentage labels with better formatting
    for i, (contribution, percentage) in enumerate(zip(contributions_df['contribution'], contributions_df['percentage'])):
        ax.text(contribution + 20, i, f"€{contribution} ({percentage:.1f}%)", va='center', fontsize=11)
    
    # Improve chart styling
    ax.set_xlabel('Cost Contribution (€)', fontsize=12)
    ax.set_title('Hardware Component Cost Breakdown', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Section 3: Component details - split into two rows instead of cramming
    st.subheader("Component Details")
    
    # Section 3A: Component cost table in its own row
    # Format the data for the table
    table_data = []
    for _, row in contributions_df.iterrows():
        table_data.append({
            "Component": f"{row['icon']} {row['feature']}",
            "Cost": f"€{row['contribution']}",
            "%": f"{row['percentage']}%"
        })
    
    # Display the formatted table
    st.table(pd.DataFrame(table_data))
    
    # Section 3B: Cost optimization tips in their own section
    st.subheader("💡 Cost Optimization Tips")
    
    # Create a stylized tips container
    st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.05); 
                padding: 20px; 
                border-radius: 10px; 
                margin-bottom: 20px;">
    """, unsafe_allow_html=True)
    
    # Find the top 2 contributing components
    top_contributors = contributions_df.head(2)['feature'].tolist()
    
    # Component-specific advice - no columns, just simple layout
    st.markdown("**Component Recommendations:**")
    
    if 'Graphics Card' in top_contributors:
        st.markdown("🎮 **GPU**: Lower-tier GPU models offer better value for non-gaming use")
        
    if 'CPU Performance' in top_contributors:
        st.markdown("⚡ **CPU**: Mid-range CPUs (rating 6-7) often provide the best performance/price ratio")
        
    if 'RAM' in top_contributors:
        st.markdown("🧠 **Memory**: 16GB RAM is sufficient for most users and offers good value")
        
    if 'Screen Resolution' in top_contributors:
        st.markdown("🔍 **Display**: Full HD resolution offers the best balance of clarity and cost")

    # General tips
    st.markdown("**General Savings Tips:**")
    st.markdown("💰 Look for seasonal sales for better pricing")
    st.markdown("📉 Consider last-generation models for significant savings")
    st.markdown("🔄 RAM and storage are usually the easiest future upgrades")
    st.markdown("🧪 Compare different brands at similar specifications")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Section 4: User feedback at the bottom (without using columns to prevent nesting issues)
    st.markdown("---")
    st.subheader("Was this breakdown helpful?")
    
    # Create a simpler layout for feedback buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("👍 Yes, very helpful", key="btn_helpful"):
            log_user_feedback("positive", config_dict)
            st.success("Thanks for your feedback! We'll keep improving our price insights.")
    
    with col2:
        if st.button("👎 Not quite accurate", key="btn_inaccurate"):
            log_user_feedback("negative", config_dict)
            st.info("Thanks for letting us know. We'll work on making our estimates more accurate.")

def extract_price_range(price_range_str):
    """
    Extract minimum and maximum prices from a price range string.
    
    Args:
        price_range_str: String containing price range (e.g., '€1000-€1500')
        
    Returns:
        Tuple of (min_price, max_price)
    """
    # Handle the €2000+ case
    if "+" in price_range_str:
        min_price = float(price_range_str.replace("€", "").replace("+", ""))
        max_price = min_price * 1.5  # Estimate max as 150% of minimum for open-ended ranges
        return min_price, max_price
    
    # Handle ranges like €1000-€1500
    parts = price_range_str.replace("€", "").split("-")
    min_price = float(parts[0])
    max_price = float(parts[1]) if len(parts) > 1 else min_price
    
    return min_price, max_price
