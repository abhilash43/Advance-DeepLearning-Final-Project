"""
Feature engineering module for credit card fraud detection.

This module handles feature creation, transformation, and selection
to improve model performance on the imbalanced fraud detection dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


class FeatureEngineer:
    """
    Handles feature engineering operations for fraud detection.
    
    Strategies include:
    - Log transformation for skewed features (Amount)
    - Interaction features between Amount and Time
    - Statistical features from PCA components
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_transformers = {}
    
    def create_amount_log_feature(self, X, amount_col='Amount'):
        """
        Create log-transformed Amount feature.
        
        Motivation:
        - Amount distribution is heavily right-skewed
        - Log transformation helps normalize distribution
        - More stable for model training
        
        Parameters:
        -----------
        X : pd.DataFrame or array-like
            Input features
        amount_col : str
            Column name for Amount
            
        Returns:
        --------
        X_with_log : pd.DataFrame
            Features with added Amount_log column
        """
        if isinstance(X, pd.DataFrame):
            X_copy = X.copy()
            X_copy['Amount_log'] = np.log1p(X_copy[amount_col])
            return X_copy
        else:
            # For array input, log transform the last column (typically Amount)
            X_copy = X.copy()
            X_copy[:, -1] = np.log1p(X_copy[:, -1])
            return X_copy
    
    def create_interaction_features(self, X):
        """
        Create interaction features between Amount and Time.
        
        Motivation:
        - Fraud patterns may depend on interactions between features
        - Example: high-value transactions at unusual times
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features (must contain Amount and Time)
            
        Returns:
        --------
        X_with_interactions : pd.DataFrame
            Features with interaction terms
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X_copy = X.copy()
        
        if 'Amount' in X.columns and 'Time' in X.columns:
            # Amount × Time interaction
            X_copy['Amount_Time_interaction'] = X_copy['Amount'] * X_copy['Time']
            
            # Amount squared (non-linear relationship)
            X_copy['Amount_squared'] = X_copy['Amount'] ** 2
            
            # Time-of-day patterns (if needed)
            # Assuming Time is seconds in a day cycle
            X_copy['Time_sin'] = np.sin(2 * np.pi * X_copy['Time'] / 86400)
            X_copy['Time_cos'] = np.cos(2 * np.pi * X_copy['Time'] / 86400)
        
        return X_copy
    
    def create_statistical_features(self, X, window_size=100):
        """
        Create statistical features from rolling windows.
        
        Motivation:
        - Transaction patterns vary temporally
        - Rolling statistics capture local patterns
        - Useful for detecting anomalies
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features with Time column
        window_size : int
            Size of rolling window
            
        Returns:
        --------
        X_with_stats : pd.DataFrame
            Features with statistical features
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X_copy = X.copy()
        X_copy = X_copy.sort_values('Time').reset_index(drop=True)
        
        if 'Amount' in X.columns:
            # Rolling statistics
            X_copy['Amount_rolling_mean'] = X_copy['Amount'].rolling(
                window=window_size, min_periods=1
            ).mean()
            X_copy['Amount_rolling_std'] = X_copy['Amount'].rolling(
                window=window_size, min_periods=1
            ).std().fillna(0)
        
        return X_copy
    
    def select_top_features(self, feature_importance, n_features=20):
        """
        Select top N features based on importance scores.
        
        Parameters:
        -----------
        feature_importance : pd.Series or dict
            Feature importance scores (name -> score)
        n_features : int, default=20
            Number of top features to select
            
        Returns:
        --------
        top_features : pd.Series or list
            Top N feature names
        """
        if isinstance(feature_importance, dict):
            feature_importance = pd.Series(feature_importance)
        
        return feature_importance.nlargest(n_features)
    
    @staticmethod
    def create_log_transformer():
        """
        Create FunctionTransformer for log transformation.
        
        Returns:
        --------
        transformer : FunctionTransformer
            Ready-to-use log transformer for pipelines
        """
        return FunctionTransformer(
            func=lambda X: np.log1p(X),
            inverse_func=lambda X: np.expm1(X),
            validate=False,
            feature_names_out='one-to-one'
        )
    
    @staticmethod
    def get_pca_feature_names(n_components=28):
        """
        Generate names for PCA features.
        
        Parameters:
        -----------
        n_components : int, default=28
            Number of PCA components
            
        Returns:
        --------
        feature_names : list
            List of PCA feature names (V1, V2, ..., V28)
        """
        return [f'V{i}' for i in range(1, n_components + 1)]
    
    def handle_outliers_iqr(self, X, columns=None, multiplier=1.5):
        """
        Handle outliers using Interquartile Range (IQR) method.
        
        Motivation:
        - Fraud transactions may have extreme values
        - IQR method is robust to outliers
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        columns : list, optional
            Columns to apply IQR. If None, apply to all numeric columns.
        multiplier : float, default=1.5
            IQR multiplier for outlier detection
            
        Returns:
        --------
        X_clipped : pd.DataFrame
            Features with outliers clipped
        outlier_indices : list
            Indices of rows with outliers
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X_copy = X.copy()
        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_indices = []
        for col in columns:
            Q1 = X_copy[col].quantile(0.25)
            Q3 = X_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Find outliers
            outliers = (X_copy[col] < lower_bound) | (X_copy[col] > upper_bound)
            outlier_indices.extend(X_copy[outliers].index.tolist())
            
            # Clip to bounds
            X_copy[col] = X_copy[col].clip(lower_bound, upper_bound)
        
        return X_copy, list(set(outlier_indices))
