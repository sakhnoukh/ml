"""
Check for RAM capacity features in the computer dataset.
"""

import pandas as pd

# Load the data
data_path = './data/db_computers_2025_processed.csv'
df = pd.read_csv(data_path)

# Look for RAM-related columns
ram_cols = [col for col in df.columns if any(term in col.lower() for term in 
               ['ram', 'memoria'])]

print("RAM-related columns found:")
for col in ram_cols:
    # Print column name and some sample values
    unique_values = df[col].unique()
    print(f"\n{col}:")
    print(f"  Sample values: {unique_values[:5]}")
    print(f"  Unique values: {len(unique_values)}")
    print(f"  Null values: {df[col].isnull().sum()}")
