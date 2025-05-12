"""
Helper module to clean price values in the computer dataset.

This module handles the conversion of price ranges in European format
to standard numeric values for modeling.
"""

import pandas as pd
import numpy as np
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_price_value(price_str):
    """
    Extract a single numeric value from a price string or range.
    
    Handles formats like:
    - "1.415,88€" -> 1415.88
    - "1.415,88€ – 2.184,17€" -> 1800.025 (average)
    
    Args:
        price_str: Price string in European format
        
    Returns:
        Extracted numeric price value
    """
    if pd.isna(price_str):
        return np.nan
    
    # Convert to string if not already
    price_str = str(price_str)
    
    # Remove non-breaking spaces and other whitespace
    price_str = price_str.replace('\xa0', ' ').strip()
    
    # Find all price numbers in the string
    # Looking for patterns like "1.234,56" or "1,234.56"
    prices = []
    
    # European format (1.234,56€)
    european_matches = re.findall(r'(\d+[\.\d]*)[\.,](\d+)', price_str)
    for whole, decimal in european_matches:
        # Remove dots in whole part (they're thousand separators) and replace comma with dot
        whole = whole.replace('.', '')
        price = float(f"{whole}.{decimal}")
        prices.append(price)
    
    # If no matches found, try to find just numbers
    if not prices:
        number_matches = re.findall(r'(\d+)', price_str)
        prices = [float(num) for num in number_matches]
    
    # If still no prices found, return NaN
    if not prices:
        return np.nan
    
    # Return the average if there are multiple prices (e.g., a range)
    return sum(prices) / len(prices)

def clean_price_column(df, price_column='Precio_Rango', new_column='Price'):
    """
    Clean the price column in the dataframe.
    
    Args:
        df: DataFrame containing the data
        price_column: Name of the column containing price strings
        new_column: Name for the new cleaned price column
        
    Returns:
        DataFrame with cleaned price column
    """
    logger.info(f"Cleaning price column: {price_column}")
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Apply the extraction function to the price column
    df_copy[new_column] = df_copy[price_column].apply(extract_price_value)
    
    # Report statistics
    non_null_count = df_copy[new_column].count()
    total_count = len(df_copy)
    logger.info(f"Extracted {non_null_count} valid prices out of {total_count} entries")
    
    if non_null_count > 0:
        logger.info(f"Price range: {df_copy[new_column].min():.2f} to {df_copy[new_column].max():.2f}")
        logger.info(f"Average price: {df_copy[new_column].mean():.2f}")
    
    return df_copy

if __name__ == "__main__":
    # Example usage
    # df = pd.read_csv("../data/db_computers_2025_raw.csv", encoding='utf-8-sig')
    # df = clean_price_column(df)
    # print(df[["Precio_Rango", "Price"]].head())
    pass
