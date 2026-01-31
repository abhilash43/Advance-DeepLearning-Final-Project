"""
Data preprocessing module for credit card fraud detection.

This module handles data loading, feature-target separation, scaling,
and pipeline construction to avoid data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Handles data preprocessing operations for fraud detection.
    
    Key principles:
    - Fit preprocessors ONLY on training data
    - Apply fitted preprocessors to validation/test data
    - Prevent data leakage through proper pipeline structure
    """
    
    def __init__(self, random_state=42):
        """
        Initialize preprocessor.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.pipeline = None
        self.feature_names = None
    
    def load_data(self, filepath):
        """
        Load data from CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataset
        """
        return pd.read_csv(filepath)
    
    def separate_features_target(self, df, target_column='Class'):
        """
        Separate features and target variable.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        target_column : str, default='Class'
            Name of target column
            
        Returns:
        --------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        return X, y
    
    def stratified_train_test_split(self, X, y, test_size=0.3, val_size=0.1):
        """
        Split data into train/validation/test sets using stratification.
        
        This ensures class distribution is maintained across all splits,
        critical for imbalanced datasets.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable
        test_size : float, default=0.3
            Proportion of data for test set
        val_size : float, default=0.1
            Proportion of remaining data for validation set
            
        Returns:
        --------
        X_train, X_val, X_test : pd.DataFrame
            Feature splits
        y_train, y_val, y_test : pd.Series
            Target splits
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, 
            random_state=self.random_state
        )
        
        # Second split: train vs val
        val_fraction = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_fraction, stratify=y_temp,
            random_state=self.random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_preprocessing_pipeline(self, numeric_features=None, 
                                     amount_feature='Amount'):
        """
        Create sklearn preprocessing pipeline with proper transformations.
        
        Pipeline design:
        - Separates V1-V28 (already scaled PCA) from Amount and Time
        - Scales Amount using RobustScaler (resistant to outliers)
        - Log-transforms Amount to handle skewness
        - Prevents data leakage by fitting only on training data
        
        Parameters:
        -----------
        numeric_features : list, optional
            List of numeric feature names. If None, inferred from data.
        amount_feature : str, default='Amount'
            Name of Amount column
            
        Returns:
        --------
        pipeline : sklearn.Pipeline
            Fitted preprocessing pipeline
        """
        if numeric_features is None:
            numeric_features = ['Time', 'Amount']
        
        # Log transformation for Amount (handles skewness)
        log_transformer = FunctionTransformer(
            func=lambda X: np.log1p(X),  # log1p handles 0 values
            validate=False,
            feature_names_out='one-to-one'
        )
        
        # Amount transformation pipeline
        amount_transformer = Pipeline(steps=[
            ('log_transform', log_transformer),
            ('robust_scaler', RobustScaler())
        ])
        
        # Columnar transformation for different features
        preprocessor = ColumnTransformer(
            transformers=[
                # V1-V28 are already scaled PCA components - no transformation needed
                ('pca_features', 'passthrough', 
                 [col for col in numeric_features if col not in [amount_feature, 'Time']]),
                # Time feature - just scaling
                ('time_scaling', RobustScaler(), ['Time']),
                # Amount feature - log transform then scale
                ('amount_transform', amount_transformer, [amount_feature])
            ],
            remainder='passthrough'
        )
        
        return preprocessor
    
    def build_full_pipeline(self, numeric_features=None):
        """
        Build complete preprocessing pipeline.
        
        Parameters:
        -----------
        numeric_features : list, optional
            List of numeric feature names
            
        Returns:
        --------
        pipeline : sklearn.Pipeline
            Complete preprocessing pipeline
        """
        preprocessor = self.create_preprocessing_pipeline(numeric_features)
        
        # Wrap in Pipeline for consistency
        pipeline = Pipeline(steps=[
            ('preprocessing', preprocessor)
        ])
        
        return pipeline
    
    def fit_transform_data(self, X_train, pipeline=None):
        """
        Fit pipeline on training data and transform.
        
        IMPORTANT: Only fit on training data to prevent data leakage.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        pipeline : sklearn.Pipeline, optional
            Pipeline to fit. If None, creates one.
            
        Returns:
        --------
        X_train_transformed : array-like
            Transformed training features
        fitted_pipeline : sklearn.Pipeline
            Fitted pipeline for later use
        """
        if pipeline is None:
            pipeline = self.build_full_pipeline(numeric_features=X_train.columns.tolist())
        
        X_train_transformed = pipeline.fit_transform(X_train)
        self.pipeline = pipeline
        self.feature_names = X_train.columns.tolist()
        
        return X_train_transformed, pipeline
    
    def transform_data(self, X, pipeline):
        """
        Transform data using fitted pipeline.
        
        Apply ONLY the fitted pipeline - never re-fit on new data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features to transform
        pipeline : sklearn.Pipeline
            Already fitted pipeline
            
        Returns:
        --------
        X_transformed : array-like
            Transformed features
        """
        return pipeline.transform(X)
    
    @staticmethod
    def get_feature_names_from_transformer(transformer, feature_input_names):
        """
        Extract feature names from fitted ColumnTransformer.
        
        Parameters:
        -----------
        transformer : sklearn.ColumnTransformer
            Fitted transformer
        feature_input_names : list
            Original feature names
            
        Returns:
        --------
        feature_names : list
            Output feature names from transformer
        """
        feature_names = []
        for name, trans, columns in transformer.transformers_:
            if trans == 'passthrough':
                feature_names.extend(columns)
            else:
                try:
                    # Try to get feature names from transformer
                    feature_names.extend(
                        trans.get_feature_names_out(columns)
                    )
                except AttributeError:
                    # If not available, use original names
                    feature_names.extend(columns)
        return feature_names
