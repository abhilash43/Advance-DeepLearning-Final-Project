"""
Machine learning models module for credit card fraud detection.

This module implements various ML models with focus on handling imbalanced data
and using appropriate evaluation metrics.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np


class ModelFactory:
    """Factory for creating and configuring ML models for fraud detection."""
    
    @staticmethod
    def create_logistic_regression(random_state=42, max_iter=1000, class_weight='balanced'):
        """
        Create Logistic Regression model.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        max_iter : int
            Maximum iterations for solver
        class_weight : str or dict
            Handle class imbalance:
            - 'balanced': weight classes inversely proportional to frequency
            
        Returns:
        --------
        model : LogisticRegression
            Configured logistic regression model
        """
        return LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            class_weight=class_weight,
            solver='lbfgs',
            n_jobs=-1
        )
    
    @staticmethod
    def create_knn(n_neighbors=5, weights='distance', metric='minkowski'):
        """
        Create K-Nearest Neighbors model.
        
        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors to use
        weights : str
            'uniform' or 'distance':
            - 'distance': closer neighbors have more influence
        metric : str
            Distance metric
            
        Returns:
        --------
        model : KNeighborsClassifier
            Configured KNN model
        """
        return KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            n_jobs=-1
        )
    
    @staticmethod
    def create_decision_tree(random_state=42, max_depth=None, 
                            min_samples_split=2, class_weight='balanced'):
        """
        Create Decision Tree model.
        
        Parameters:
        -----------
        random_state : int
            Random seed
        max_depth : int or None
            Maximum tree depth (None = unlimited)
        min_samples_split : int
            Minimum samples required to split a node
        class_weight : str or dict
            Handle class imbalance
            
        Returns:
        --------
        model : DecisionTreeClassifier
            Configured decision tree model
        """
        return DecisionTreeClassifier(
            random_state=random_state,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight
        )
    
    @staticmethod
    def create_random_forest(n_estimators=100, random_state=42,
                            max_depth=None, min_samples_split=2,
                            class_weight='balanced'):
        """
        Create Random Forest model.
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in forest
        random_state : int
            Random seed
        max_depth : int or None
            Maximum tree depth
        min_samples_split : int
            Minimum samples for split
        class_weight : str or dict
            Handle class imbalance
            
        Returns:
        --------
        model : RandomForestClassifier
            Configured random forest model
        """
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            n_jobs=-1
        )
    
    @staticmethod
    def create_svm(kernel='rbf', C=1.0, class_weight='balanced', random_state=42):
        """
        Create Support Vector Machine model.
        
        Parameters:
        -----------
        kernel : str
            Kernel type: 'linear', 'rbf', 'poly', 'sigmoid'
        C : float
            Regularization parameter (inverse of regularization strength)
        class_weight : str or dict
            Handle class imbalance
        random_state : int
            Random seed
            
        Returns:
        --------
        model : SVC
            Configured SVM model
        """
        return SVC(
            kernel=kernel,
            C=C,
            class_weight=class_weight,
            probability=True,  # Enable probability estimates
            random_state=random_state
        )
    
    @staticmethod
    def get_all_models(random_state=42):
        """
        Create all models with default configuration.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        models : dict
            Dictionary of model name -> model instance
        """
        models = {
            'Logistic Regression': ModelFactory.create_logistic_regression(
                random_state=random_state
            ),
            'KNN (k=5)': ModelFactory.create_knn(n_neighbors=5),
            'Decision Tree': ModelFactory.create_decision_tree(
                random_state=random_state,
                max_depth=10
            ),
            'Random Forest': ModelFactory.create_random_forest(
                n_estimators=100,
                random_state=random_state,
                max_depth=15
            ),
            'SVM (RBF)': ModelFactory.create_svm(
                kernel='rbf',
                C=1.0,
                random_state=random_state
            )
        }
        return models
