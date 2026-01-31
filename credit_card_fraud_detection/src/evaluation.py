"""
Model evaluation module for credit card fraud detection.

This module handles model evaluation with metrics appropriate for imbalanced
classification problems.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, auc, f1_score, precision_score,
    recall_score, accuracy_score, matthews_corrcoef
)


class ModelEvaluator:
    """
    Handles model evaluation and metrics computation for imbalanced classification.
    
    Key metrics for imbalanced data:
    - Recall (Sensitivity): True Positive Rate (critical for fraud detection)
    - Precision: Positive Predictive Value (cost of false alarms)
    - F1-Score: Harmonic mean of precision and recall
    - ROC-AUC: Threshold-independent performance measure
    - PR-AUC: Precision-Recall AUC (better for imbalanced data)
    """
    
    @staticmethod
    def compute_metrics(y_true, y_pred, y_pred_proba=None, average='binary'):
        """
        Compute comprehensive evaluation metrics.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_pred_proba : array-like, optional
            Predicted probabilities for positive class
        average : str
            Averaging method ('binary' for binary classification)
            
        Returns:
        --------
        metrics : dict
            Dictionary of metrics
        """
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred),
            'MCC': matthews_corrcoef(y_true, y_pred),
        }
        
        # ROC-AUC requires probabilities
        if y_pred_proba is not None:
            metrics['ROC-AUC'] = roc_auc_score(y_true, y_pred_proba)
            
            # PR-AUC (often better for imbalanced data)
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['PR-AUC'] = auc(recall, precision)
        
        return metrics
    
    @staticmethod
    def get_classification_report(y_true, y_pred):
        """
        Get detailed classification report.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
            
        Returns:
        --------
        report : str
            Formatted classification report
        """
        return classification_report(
            y_true, y_pred,
            target_names=['Normal', 'Fraud'],
            digits=4
        )
    
    @staticmethod
    def get_confusion_matrix(y_true, y_pred):
        """
        Get confusion matrix.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
            
        Returns:
        --------
        cm : ndarray
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def get_confusion_matrix_dict(y_true, y_pred):
        """
        Get confusion matrix as interpretable dictionary.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
            
        Returns:
        --------
        cm_dict : dict
            Named confusion matrix components
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'TP': tp,  # True Positives (correctly detected fraud)
            'FP': fp,  # False Positives (normal flagged as fraud)
            'FN': fn,  # False Negatives (fraud missed - BAD!)
            'TN': tn   # True Negatives (correctly identified normal)
        }
    
    @staticmethod
    def get_roc_curve(y_true, y_pred_proba):
        """
        Get ROC curve data.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities
            
        Returns:
        --------
        fpr : ndarray
            False positive rates
        tpr : ndarray
            True positive rates
        auc_score : float
            ROC-AUC score
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        return fpr, tpr, auc_score
    
    @staticmethod
    def get_precision_recall_curve(y_true, y_pred_proba):
        """
        Get Precision-Recall curve data.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities
            
        Returns:
        --------
        precision : ndarray
            Precision values
        recall : ndarray
            Recall values
        pr_auc : float
            PR-AUC score
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        return precision, recall, pr_auc
    
    @staticmethod
    def create_metrics_dataframe(models_results):
        """
        Create a pandas DataFrame from model results for easy comparison.
        
        Parameters:
        -----------
        models_results : dict
            Dictionary: model_name -> metrics_dict
            
        Returns:
        --------
        df : pd.DataFrame
            Models comparison table
        """
        df = pd.DataFrame(models_results).T
        df = df.round(4)
        return df.sort_values('F1-Score', ascending=False)
    
    @staticmethod
    def print_model_comparison(models_results):
        """
        Print formatted model comparison table.
        
        Parameters:
        -----------
        models_results : dict
            Dictionary: model_name -> metrics_dict
        """
        df = ModelEvaluator.create_metrics_dataframe(models_results)
        print('\n' + '='*100)
        print('MODEL PERFORMANCE COMPARISON (Test Set)')
        print('='*100)
        print(df.to_string())
        print('='*100)
    
    @staticmethod
    def get_best_model(models_results, metric='F1-Score'):
        """
        Get best model based on specified metric.
        
        Parameters:
        -----------
        models_results : dict
            Dictionary: model_name -> metrics_dict
        metric : str
            Metric to use for ranking
            
        Returns:
        --------
        best_model : str
            Name of best model
        best_score : float
            Score of best model
        """
        scores = {name: results.get(metric, 0) 
                 for name, results in models_results.items()}
        best_model = max(scores, key=scores.get)
        best_score = scores[best_model]
        return best_model, best_score
