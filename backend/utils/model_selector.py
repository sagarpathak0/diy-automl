import numpy as np
import pandas as pd

def select_model_type(X, y, problem_type):
    """
    Select appropriate model based on data characteristics:
    - Data size
    - Number of features
    - Problem type (classification or regression)
    - Feature correlations
    - Class distribution (for classification)

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        problem_type (str): 'classification' or 'regression'
    
    Returns:
        str: The name of the selected model type
    """
    n_samples, n_features = X.shape
    
    # Check data size
    small_data = n_samples < 1000
    large_data = n_samples > 10000
    
    # Check feature count
    few_features = n_features < 10
    many_features = n_features > 50
    
    if problem_type == 'classification':
        # Get class distribution
        class_counts = np.bincount(y.astype(int))
        n_classes = len(class_counts)
        
        # Check for class imbalance
        class_ratio = np.min(class_counts) / np.max(class_counts) if np.max(class_counts) > 0 else 0
        balanced_classes = class_ratio > 0.3
        
        # Check if binary or multiclass
        binary = n_classes <= 2
        
        # Simple decision rules for classification
        if small_data:
            if binary and few_features:
                return 'logistic_regression'
            elif few_features and balanced_classes:
                return 'decision_tree'
            else:
                return 'random_forest'
        elif large_data:
            if many_features:
                return 'random_forest'
            elif binary:
                return 'svm'
            else:
                return 'neural_network'
        else:  # Medium data size
            if binary and few_features:
                return 'logistic_regression'
            elif few_features:
                return 'svm'
            else:
                return 'random_forest'
    
    else:  # Regression
        # Check feature correlations
        if few_features:
            try:
                # Calculate correlation matrix
                corr_matrix = X.corr().abs()
                # Calculate mean correlation (excluding self-correlations)
                mean_corr = (corr_matrix.sum().sum() - n_features) / (n_features * (n_features - 1))
                linear_relationship = mean_corr > 0.3
            except:
                linear_relationship = False
        else:
            linear_relationship = False
        
        # Simple decision rules for regression
        if small_data:
            if few_features and linear_relationship:
                return 'linear_regression'
            else:
                return 'decision_tree'
        elif large_data:
            if many_features:
                return 'random_forest'
            else:
                return 'neural_network'
        else:  # Medium data size
            if few_features and linear_relationship:
                return 'linear_regression'
            else:
                return 'random_forest'