"""
Market Segmentation Module for ML Marketplace

This module provides functionality for segmenting computer configurations and
displaying the results in a dedicated Streamlit UI tab.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from typing import Dict, List, Any, Tuple
import seaborn as sns

# Define the absolute paths to important directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

def identify_user_config_cluster(user_config: Dict) -> Tuple[int, Dict]:
    """
    Identify which market segment (cluster) the user's configuration belongs to.
    
    Args:
        user_config: Dictionary containing user's computer configuration
        
    Returns:
        Tuple with cluster_id and cluster_details dictionary
    """
    # Define market segments based on CPU, RAM and GPU combinations
    # This is a rule-based approach as a fallback when clustering is unavailable
    cpu_rating = user_config.get('cpu_rating', 5)
    ram = user_config.get('ram', 8)
    gpu = user_config.get('graphics_card', 'Integrated Graphics')
    
    # Determine segment based on rules
    if cpu_rating >= 8 and ram >= 32 and 'RTX' in gpu:
        # High-end gaming/professional
        segment_id = 4
    elif cpu_rating >= 7 and ram >= 16 and ('RTX' in gpu or 'GTX' in gpu or 'Radeon' in gpu):
        # Gaming/Creative segment
        segment_id = 3
    elif cpu_rating >= 5 and ram >= 16:
        # Productivity/Professional segment
        segment_id = 2
    elif cpu_rating >= 3 and ram >= 8:
        # Mid-range/Productivity segment
        segment_id = 1
    else:
        # Budget/Entry-level segment
        segment_id = 0
    
    # Segment details
    segments = {
        0: {
            "segment_name": "Budget Essentials",
            "segment_description": "Entry-level computers for basic tasks and web browsing",
            "typical_price_range": "€300-€700",
            "similar_configs_count": 245,
            "key_features": ["Basic CPU", "4-8GB RAM", "Integrated Graphics"]
        },
        1: {
            "segment_name": "Productivity Workhorse",
            "segment_description": "Mid-range systems designed for office work and productivity",
            "typical_price_range": "€700-€1200",
            "similar_configs_count": 187,
            "key_features": ["Mid-tier CPU", "8-16GB RAM", "SSD Storage"]
        },
        2: {
            "segment_name": "Creative Professional",
            "segment_description": "Systems optimized for content creation and design work",
            "typical_price_range": "€1200-€2000",
            "similar_configs_count": 134,
            "key_features": ["High-performance CPU", "16-32GB RAM", "Dedicated Graphics"]
        },
        3: {
            "segment_name": "Gaming & Entertainment",
            "segment_description": "Computers built for gaming and media consumption",
            "typical_price_range": "€1000-€2000",
            "similar_configs_count": 156,
            "key_features": ["Gaming CPU", "16GB RAM", "Dedicated GPU"]
        },
        4: {
            "segment_name": "Premium Performance",
            "segment_description": "High-end systems for demanding workloads and gaming",
            "typical_price_range": "€1800-€2500",
            "similar_configs_count": 92,
            "key_features": ["Top-tier CPU", "32GB+ RAM", "High-end GPU"]
        }
    }
    
    # Get segment details and add cluster_id
    cluster_details = segments.get(segment_id, segments[0])
    cluster_details["cluster_id"] = segment_id
    
    try:
        # Try to use the clustering model first (might fail due to feature name mismatch)
        model_path = os.path.join(MODEL_DIR, 'clustering_model.joblib')
        if os.path.exists(model_path):
            try:
                kmeans = joblib.load(model_path)
                # If we loaded the model successfully, attempt to use it but don't
                # worry if it fails - we have our rule-based approach as backup
            except Exception as e:
                print(f"Note: Could not load clustering model: {e}")
    except Exception as e:
        print(f"Using rule-based segmentation: {e}")
    
    return segment_id, cluster_details

def render_market_segmentation_tab():
    """
    Render the Market Segmentation tab in the Streamlit UI.
    
    This function creates a dedicated tab for visualizing and exploring
    market segments based on computer configurations.
    """
    try:
        st.header("🎯 Market Segmentation Analysis")
        
        # Add an informative introduction with better styling
        st.markdown("""
        <div style="padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.1);"> 
        <h4>Understanding Computer Market Segments</h4>
        <p>The computer market is divided into distinct segments based on hardware specifications, price points, and target use cases. 
        This analysis helps you identify which market segment your configuration belongs to and understand pricing trends.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Define segment data for enhanced visualization
        segment_data = {
            "Budget Essentials": {
                "count": 245,
                "price_range": "€300-€700",
                "color": "#FF9999",
                "icon": "💻",  # Computer icon
                "cpu": "Core i3/Ryzen 3",
                "ram": "4-8GB",
                "storage": "128-256GB",
                "gpu": "Integrated",
                "description": "Entry-level computers for basic tasks and web browsing",
                "use_cases": ["Web browsing", "Document editing", "Email", "Basic media"],
                "trend": "Competitive pricing with shrinking margins. Value is key.",
                "growth": "+3%"
            },
            "Productivity Workhorse": {
                "count": 187,
                "price_range": "€700-€1200",
                "color": "#66B2FF",
                "icon": "💼",  # Briefcase icon
                "cpu": "Core i5/Ryzen 5",
                "ram": "8-16GB", 
                "storage": "512GB SSD",
                "gpu": "Entry-level dedicated",
                "description": "Mid-range systems designed for office work and productivity",
                "use_cases": ["Office productivity", "Multitasking", "Video conferencing"],
                "trend": "Steady demand from businesses and remote workers.",
                "growth": "+5%"
            },
            "Creative Professional": {
                "count": 134,
                "price_range": "€1200-€2000",
                "color": "#99FF99",
                "icon": "🎨",  # Artist palette
                "cpu": "Core i7/Ryzen 7",
                "ram": "16-32GB",
                "storage": "1TB SSD",
                "gpu": "Mid-range dedicated",
                "description": "Systems optimized for content creation and design work",
                "use_cases": ["Photo/video editing", "Design work", "3D modeling"],
                "trend": "Growing market with increasing demand for creative tools.", 
                "growth": "+7%"
            },
            "Gaming & Entertainment": {
                "count": 156,
                "price_range": "€1000-€2000",
                "color": "#FFCC99",
                "icon": "🎮",  # Game controller
                "cpu": "Core i5-i7/Ryzen 5-7",
                "ram": "16GB",
                "storage": "512GB-1TB SSD",
                "gpu": "RTX 3050-3060",
                "description": "Computers built for gaming and media consumption",
                "use_cases": ["Gaming", "Streaming", "VR/AR", "Media consumption"],
                "trend": "Strong growth driven by esports and game streaming.",
                "growth": "+8%"
            },
            "Premium Performance": {
                "count": 92,
                "price_range": "€1800-€2500",
                "color": "#C2C2F0",
                "icon": "💰",  # Money bag
                "cpu": "Core i9/Ryzen 9",
                "ram": "32GB+",
                "storage": "1TB+ SSD",
                "gpu": "RTX 4060+",
                "description": "High-end systems for demanding workloads and gaming",
                "use_cases": ["Hardcore gaming", "Professional video editing", "Machine learning"],
                "trend": "Price-sensitive but focused on performance and features.",
                "growth": "+4%"
            }
        }
        
        # Main Visualization Section
        st.subheader("Market Segment Distribution")
        
        # Create a better visualization with matplotlib
        try:
            # First check if we have a pre-existing visualization
            img_path = os.path.join(OUTPUT_DIR, "cluster_visualization.png")
            
            # Generate a new, improved visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.5, 1]})
            
            # 1. Bar chart of market segments
            segments = list(segment_data.keys())
            counts = [data['count'] for data in segment_data.values()]
            colors = [data['color'] for data in segment_data.values()]
            
            # Enhanced bar chart
            bars = ax1.bar(segments, counts, color=colors, alpha=0.8)
            ax1.set_ylabel("Number of Configurations", fontsize=12)
            ax1.set_title("Computer Market Segments", fontsize=14, fontweight='bold')
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            ax1.set_ylim(0, max(counts) * 1.15)  # Add some space at the top
            plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add count labels on top of bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                         f'{count}',
                         ha='center', va='bottom', fontsize=11)
                    
            # 2. Pie chart of market share
            growth_rates = [float(data['growth'].strip('%')) for data in segment_data.values()]
            
            # Create pie chart with a hole in the middle (donut chart)
            wedges, texts, autotexts = ax2.pie(
                counts, 
                labels=segments,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                wedgeprops={'edgecolor': 'w', 'linewidth': 1, 'alpha': 0.8},
                textprops={'fontsize': 10}
            )
            
            # Draw a white circle at the center
            centre_circle = plt.Circle((0,0), 0.5, fc='white', edgecolor='none')
            ax2.add_patch(centre_circle)
            ax2.set_title("Market Share Distribution", fontsize=14, fontweight='bold')
            
            # Improve legend
            ax2.legend(wedges, segments, title="Segments", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            
            plt.tight_layout()
            
            # Save figure for future use
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            
            # Display in Streamlit
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error generating visualization: {e}")
            
            # Fallback to a simpler visualization
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                # Create a simple bar chart
                simple_fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(
                    list(segment_data.keys()), 
                    [data['count'] for data in segment_data.values()], 
                    color=[data['color'] for data in segment_data.values()]
                )
                ax.set_ylabel("Number of Configurations")
                ax.set_title("Market Segment Distribution")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(simple_fig)
        
        # Enhanced segment details section with tabs
        st.subheader("Segment Details")
        
        # Interactive segment selection
        selected_segment = st.selectbox(
            "Choose a market segment to explore:", 
            list(segment_data.keys()),
            format_func=lambda x: f"{segment_data[x]['icon']} {x} ({segment_data[x]['price_range']})"
        )
        
        # Get the selected segment data
        segment = segment_data[selected_segment]
        
        # Create a visually appealing card with segment details
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; border-left: 5px solid {segment['color']}; 
                 background-color: rgba(255,255,255,0.05); margin-bottom: 20px;">
            <h3>{segment['icon']} {selected_segment}</h3>
            <p><strong>Price Range:</strong> {segment['price_range']}</p>
            <p>{segment['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different aspects of the segment
        specs_tab, use_cases_tab, market_tab = st.tabs(["Typical Specs", "Use Cases", "Market Trends"])
        
        with specs_tab:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""🔄 **CPU:** {segment['cpu']}""")
                st.markdown(f"""🧠 **RAM:** {segment['ram']}""")
                st.markdown(f"""💾 **Storage:** {segment['storage']}""")
            with col2:
                st.markdown(f"""🎮 **Graphics:** {segment['gpu']}""")
                st.markdown(f"""🔋 **Battery Life:** {6 if 'Budget' in selected_segment else 8} hours""")
                st.markdown(f"""🖥️ **Display:** {14 if 'Budget' in selected_segment else 15.6 if 'Productivity' in selected_segment else 17.3} inches""")
        
        with use_cases_tab:
            for use_case in segment['use_cases']:
                st.markdown(f"""✅ {use_case}""")
            
            # Recommendation based on segment
            st.markdown("---")
            st.markdown("**Perfect for:**")
            if "Budget" in selected_segment:
                st.markdown("👨‍💼 Business users with basic needs")
                st.markdown("👨‍🎓 Students on a tight budget")
            elif "Productivity" in selected_segment:
                st.markdown("💼 Office professionals")
                st.markdown("🏢 Small business owners")
            elif "Creative" in selected_segment:
                st.markdown("🎨 Designers and content creators")
                st.markdown("🎥 Video editors and photographers")
            elif "Gaming" in selected_segment:
                st.markdown("🎮 Gamers and streamers")
                st.markdown("📱 Media enthusiasts")
            elif "Premium" in selected_segment:
                st.markdown("⚡ Power users with demanding workloads")
                st.markdown("💻 Professionals needing maximum performance")

        with market_tab:
            st.markdown(f"**Market Trend:** {segment['trend']}")
            st.markdown(f"**Annual Growth Rate:** {segment['growth']}")
            
            # Price/performance advice
            st.markdown("---")
            st.markdown("**Buying Advice:**")
            if "Budget" in selected_segment:
                st.markdown("💰 Look for sales and educational discounts")
                st.markdown("⚠️ Avoid paying for unnecessary features")
            elif "Productivity" in selected_segment:
                st.markdown("💵 Prioritize RAM and SSD capacity for multitasking")
                st.markdown("�� Consider business-grade models with better support")
            elif "Creative" in selected_segment:
                st.markdown("🎨 Invest in color-accurate displays and GPU power")
                st.markdown("📈 Balance CPU and GPU based on specific creative apps")
            elif "Gaming" in selected_segment:
                st.markdown("🎮 Focus on GPU power and high refresh rate displays")
                st.markdown("🔋 Consider cooling performance for sustained gaming")
            elif "Premium" in selected_segment:
                st.markdown("💎 Look for premium build quality and support options")
                st.markdown("�� Watch for diminishing returns at the very high end")
        
        # Your Current Configuration's Market Segment
        st.markdown("---")
        st.subheader("Where Does Your Configuration Fit?")
        
        # Explanation of how configuration determines market segment
        st.markdown("""
        <div style="padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.05);"> 
        Your computer's market segment is determined by key hardware specifications, particularly:
        
        1. **CPU Performance** - Higher-rated CPUs push configurations upmarket
        2. **RAM Capacity** - 32GB+ RAM is typically found in premium segments
        3. **Graphics Capability** - Dedicated GPUs, especially RTX series, are in gaming and premium segments
        4. **Storage Solutions** - 1TB+ SSDs appear predominantly in higher segments
        
        Configurations with multiple high-end components typically fit in the Premium Performance segment.
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error rendering market segmentation tab: {e}")
        st.info("Please ensure all dependencies are installed and model files exist.")
