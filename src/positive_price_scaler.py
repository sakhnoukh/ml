"""
Positive Price Scaler module for the ML Marketplace.

This module provides an enhanced price scaler that guarantees positive price predictions.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class PositivePriceScaler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for scaling price values during training and
    correctly inverse-transforming them to positive values during prediction.
    """
    
    def __init__(self, price_column='Price', min_price=100):
        """
        Initialize the positive price scaler.
        
        Args:
            price_column: Name of the price column
            min_price: Minimum allowable price in the output
        """
        self.price_column = price_column
        self.min_price = min_price
        self.price_mean_ = None
        self.price_std_ = None
    
    def fit(self, X, y=None):
        """
        Fit the scaler by computing mean and standard deviation of price column.
        
        Args:
            X: DataFrame containing the price column
            y: Ignored (included for compatibility)
            
        Returns:
            self
        """
        if isinstance(X, pd.DataFrame) and self.price_column in X.columns:
            self.price_mean_ = X[self.price_column].mean()
            self.price_std_ = X[self.price_column].std()
        else:
            # If X is not a DataFrame or doesn't have the price column,
            # we're probably in prediction mode, so do nothing
            pass
        
        return self
    
    def transform(self, X):
        """
        Transform the price column by standardizing it.
        
        Args:
            X: DataFrame containing the price column
            
        Returns:
            Transformed DataFrame
        """
        X_copy = X.copy()
        
        if isinstance(X, pd.DataFrame) and self.price_column in X.columns and self.price_mean_ is not None:
            X_copy[self.price_column] = (X_copy[self.price_column] - self.price_mean_) / self.price_std_
        
        return X_copy
    
    def transform_target(self, price_values):
        """
        Transform price values by standardizing them.
        
        Args:
            price_values: Series or array of price values to scale
            
        Returns:
            Scaled price values
        """
        if self.price_mean_ is not None and self.price_std_ is not None:
            return (price_values - self.price_mean_) / self.price_std_
        else:
            return price_values  # Return as is if not fitted
    
    def inverse_transform(self, X, y_pred):
        """
        Ensure predictions are always positive and reasonable.
        
        Args:
            X: Original input features (not used, included for compatibility)
            y_pred: Predicted price values in standardized scale
            
        Returns:
            Predicted price values in original scale, always positive
        """
        if self.price_mean_ is not None and self.price_std_ is not None:
            # Calculate raw price
            raw_price = y_pred * self.price_std_ + self.price_mean_
            
            # Handle both scalar and array/Series inputs
            if hasattr(raw_price, '__iter__'):
                # For arrays/Series, use element-wise operations
                return np.maximum(np.abs(raw_price), self.min_price)
            else:
                # For scalar values
                return max(abs(raw_price), self.min_price)
        else:
            # Similar handling for y_pred
            if hasattr(y_pred, '__iter__'):
                return np.maximum(np.abs(y_pred), self.min_price)
            else:
                return max(abs(y_pred), self.min_price)
