import numpy as np
import pandas as pd

def engineer_features(X_train, X_test, problem_type):
    """
    Apply feature engineering to the training and test data.
    This includes:
    - Polynomial features for regression problems
    - Interaction terms between important features
    - Feature scaling already handled in data_processing.py
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        problem_type (str): 'classification' or 'regression'
        
    Returns:
        Tuple containing engineered X_train and X_test
    """
    # Ensure both dataframes have the same columns
    combined_columns = set(X_train.columns) | set(X_test.columns)
    for col in combined_columns:
        if col not in X_train.columns:
            X_train[col] = 0
        if col not in X_test.columns:
            X_test[col] = 0
    
    # Reorder columns to match
    X_test = X_test[X_train.columns]
    
    # Create polynomial features for regression problems
    if problem_type == 'regression' and X_train.shape[1] <= 10:  # Only for smaller feature sets
        X_train, X_test = add_polynomial_features(X_train, X_test, degree=2)
    
    # Create interaction terms between important features
    X_train, X_test = add_interaction_terms(X_train, X_test)
    
    # Add sum and mean of numerical features as new features
    X_train, X_test = add_aggregate_features(X_train, X_test)
    
    return X_train, X_test

def add_polynomial_features(X_train, X_test, degree=2):
    """Add polynomial features of specified degree."""
    # Only use top numerical features to avoid explosion in feature count
    num_cols = X_train.select_dtypes(include=np.number).columns
    if len(num_cols) > 5:  # Limit to top 5 numerical features
        # Use variance as a simple heuristic for importance
        variances = X_train[num_cols].var()
        top_cols = variances.nlargest(5).index
    else:
        top_cols = num_cols
    
    # Create polynomial features
    for col in top_cols:
        for d in range(2, degree + 1):
            X_train[f"{col}^{d}"] = X_train[col] ** d
            X_test[f"{col}^{d}"] = X_test[col] ** d
    
    return X_train, X_test

def add_interaction_terms(X_train, X_test):
    """Add interaction terms between important features."""
    # Only use top numerical features
    num_cols = X_train.select_dtypes(include=np.number).columns
    if len(num_cols) > 5:
        # Use correlation with other features as a simple heuristic for importance
        corr_matrix = X_train[num_cols].corr().abs()
        # Sum correlations for each feature
        importance = corr_matrix.sum()
        top_cols = importance.nlargest(5).index
    else:
        top_cols = num_cols
    
    # Create interactions between pairs of top features
    for i in range(len(top_cols)):
        for j in range(i+1, len(top_cols)):
            col_i = top_cols[i]
            col_j = top_cols[j]
            interaction_name = f"{col_i}*{col_j}"
            X_train[interaction_name] = X_train[col_i] * X_train[col_j]
            X_test[interaction_name] = X_test[col_i] * X_test[col_j]
    
    return X_train, X_test

def add_aggregate_features(X_train, X_test):
    """Add sum and mean of numerical features as new features."""
    num_cols = X_train.select_dtypes(include=np.number).columns
    
    if len(num_cols) >= 3:  # Only add if we have enough numerical features
        X_train['sum_num_features'] = X_train[num_cols].sum(axis=1)
        X_train['mean_num_features'] = X_train[num_cols].mean(axis=1)
        
        X_test['sum_num_features'] = X_test[num_cols].sum(axis=1)
        X_test['mean_num_features'] = X_test[num_cols].mean(axis=1)
    
    return X_train, X_test