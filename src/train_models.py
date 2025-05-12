"""
Training script for the ML Marketplace application.

This script orchestrates the end-to-end pipeline for training and saving the models:
1. Data preprocessing
2. Feature engineering
3. Model training
4. Similarity model creation

Usage:
    python train_models.py [--data-path PATH] [--models-path PATH]
"""

import os
import argparse
import logging
import sys
import importlib
import joblib

# Helper function to check if a module is available
def is_module_available(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

# Import essential modules
import pandas as pd
import numpy as np

# Conditionally import optional modules
if is_module_available('matplotlib.pyplot'):
    import matplotlib.pyplot as plt
else:
    plt = None
    print("Warning: matplotlib not available. Visualizations will be disabled.")

# Import local modules
from preprocessing import load_data, preprocess_data
from features import engineer_features, identify_core_features, get_feature_names, save_feature_pipeline
from clean_price import clean_price_column
from similarity import create_similarity_model
from price_scaler import PriceScaler

# Conditionally import model and similarity modules
try:
    from models import train_and_save_model
except ImportError:
    print("Warning: models module could not be imported. Model training will be disabled.")
    train_and_save_model = None

try:
    from similarity import create_similarity_model
except ImportError:
    print("Warning: similarity module could not be imported. Similarity model creation will be disabled.")
    create_similarity_model = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ML Marketplace models")
    parser.add_argument("--data-path", type=str, default="data",
                      help="Path to the data directory")
    parser.add_argument("--models-path", type=str, default="models",
                      help="Path to save the trained models")
    return parser.parse_args()


def main():
    """Run the complete training pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Define file paths
    raw_data_file = os.path.join(args.data_path, "db_computers_2025_raw.csv")
    processed_data_file = os.path.join(args.data_path, "db_computers_2025_processed.csv")
    feature_pipeline_file = os.path.join(args.models_path, "feature_pipeline.joblib")
    price_model_file = os.path.join(args.models_path, "price_model.joblib")
    similarity_model_file = os.path.join(args.models_path, "similarity_model.joblib")
    shap_plot_file = os.path.join(args.models_path, "shap_summary.png")
    
    # Create directories if they don't exist
    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.models_path, exist_ok=True)
    
    # Step 1: Data Preprocessing
    logger.info("Step 1: Data Preprocessing")
    
    # Preprocess data
    try:
        logger.info("Preprocessing data")
        df = preprocess_data(raw_data_file)
        logger.info(f"Data preprocessed. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        return
        
    # Clean data
    try:
        
        # Clean price column
        logger.info("Cleaning price column")
        if 'Precio_Rango' in df.columns:
            df = clean_price_column(df, price_column='Precio_Rango', new_column='Price')
            logger.info(f"Price column cleaned and 'Price' column created")
        else:
            logger.warning("'Precio_Rango' column not found. Skipping price cleaning.")
            
        logger.info(f"Preprocessing complete. Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Saving preprocessed data to {processed_data_file}")
        df.to_csv(processed_data_file, index=False)
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        return
    
    # Step 2: Feature Engineering
    logger.info("Step 2: Feature Engineering")
    
    # Identify core features
    numeric_features, categorical_features = identify_core_features(df)
    logger.info(f"Identified {len(numeric_features)} numeric features: {numeric_features}")
    logger.info(f"Identified {len(categorical_features)} categorical features: {categorical_features}")
    
    # Engineer features
    logger.info("Engineering features")
    X_transformed, feature_pipeline = engineer_features(
        df,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )
    
    # Get feature names
    feature_names = get_feature_names(feature_pipeline, numeric_features, categorical_features)
    
    # Save feature pipeline
    logger.info(f"Saving feature pipeline to {feature_pipeline_file}")
    save_feature_pipeline(feature_pipeline, feature_pipeline_file)
    
    # Apply price scaling and save scaler
    price_scaler = PriceScaler(price_column='Price')
    price_scaler.fit(df)
    price_scaler_file = os.path.join(args.models_path, 'price_scaler.joblib')
    joblib.dump(price_scaler, price_scaler_file)
    logger.info(f"Saved price scaler to {price_scaler_file}")
    logger.info(f"Price statistics - Mean: {price_scaler.price_mean_:.2f}, Std Dev: {price_scaler.price_std_:.2f}")
    
    # Scale the price values for training
    scaled_prices = price_scaler.transform_target(df['Price'])
    logger.info(f"Scaled prices for training - Min: {scaled_prices.min():.2f}, Max: {scaled_prices.max():.2f}, Mean: {scaled_prices.mean():.2f}")
    
    # Step 3: Model Training (if available)
    if train_and_save_model is not None and 'Price' in df.columns:
        logger.info("Step 3: Model Training")
        
        try:
            # Train and save model
            logger.info(f"Training price prediction model")
            best_model, results = train_and_save_model(
                X_transformed, 
                scaled_prices,
                feature_names=feature_names,
                model_path=price_model_file,
                shap_plot_path=shap_plot_file
            )
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            logger.info("Continuing with next steps...")
    else:
        if 'Price' not in df.columns:
            logger.error("'Price' column not found in the dataset. Cannot train price prediction model.")
        else:
            logger.warning("Model training skipped due to missing dependencies.")
    
    # Step 4: Similarity Model (if available)
    if create_similarity_model is not None:
        logger.info("Step 4: Similarity Model")
        
        try:
            # Create and save similarity model
            logger.info(f"Creating similarity model")
            similarity_model = create_similarity_model(
                X_transformed, 
                df,
                feature_names=feature_names,
                metric='euclidean',
                model_path=similarity_model_file
            )
            logger.info("Similarity model created successfully")
        except Exception as e:
            logger.error(f"Error creating similarity model: {str(e)}")
    else:
        logger.warning("Similarity model creation skipped due to missing dependencies.")
    
    logger.info("Training pipeline steps completed!")
    logger.info(f"Processed data saved to {processed_data_file}")
    if os.path.exists(feature_pipeline_file):
        logger.info(f"Feature pipeline saved to {feature_pipeline_file}")
    if os.path.exists(price_model_file):
        logger.info(f"Price model saved to {price_model_file}")
    if os.path.exists(similarity_model_file):
        logger.info(f"Similarity model saved to {similarity_model_file}")
        
    logger.info("\nNext steps:")
    logger.info("1. If any steps were skipped due to missing dependencies, install them and run again")
    logger.info("2. Run the Streamlit app: streamlit run src/app.py")


if __name__ == "__main__":
    main()
