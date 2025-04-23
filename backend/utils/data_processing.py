import numpy as np
import pandas as pd

def preprocess_data(data, is_training=True, target_column=None, encoders=None):
    """
    Preprocess the data for machine learning:
    - Identify numerical and categorical features
    - Handle missing values
    - Encode categorical variables
    - Normalize numerical features
    - Identify target column in training data
    
    Args:
        data (pd.DataFrame): Input data
        is_training (bool): Whether this is training data or prediction data
        target_column (str): The name of the target column if known (for prediction data)
        encoders (dict): Dictionary of category encoders to use for prediction data
    
    Returns:
        If is_training=True: X, y, target_column_name, problem_type, encoders
        If is_training=False: X_preprocessed
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Select target column for training data
    if is_training:
        y, target_column, problem_type = select_target_column(df)
        X = df.drop(target_column, axis=1)
        encoders = {}  # Initialize encoders dictionary
    else:
        if target_column is not None and target_column in df.columns:
            X = df.drop(target_column, axis=1)
        else:
            X = df
    
    # Store original column order
    original_columns = list(X.columns)
    
    # Identify numerical and categorical features
    cat_columns = X.select_dtypes(include=['object', 'category']).columns
    num_columns = X.select_dtypes(include=np.number).columns
    
    # Handle categorical features
    if is_training:
        X, encoders = encode_categorical_features(X, cat_columns, None)
    else:
        X = encode_categorical_features(X, cat_columns, encoders)
    
    # Normalize numerical features
    X = normalize_numerical_features(X, num_columns)
    
    # Ensure consistent column ordering between training and prediction
    if not is_training and encoders and 'column_order' in encoders:
        # Add missing columns with default values
        for col in encoders['column_order']:
            if col not in X.columns:
                X[col] = 0
        
        # Select only columns that were in training and in correct order
        X = X[encoders['column_order']]
    
    if is_training:
        # Store the column order for later use
        encoders['column_order'] = list(X.columns)
        return X, y, target_column, problem_type, encoders
    else:
        return X

def handle_missing_values(df):
    """Fill missing values in the dataframe."""
    # For numerical columns, fill with median
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # For categorical columns, fill with mode
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
    
    return df

def select_target_column(df):
    """
    Heuristically select the most likely target column from the dataframe.
    For simplicity, we'll use the last column as the target.
    
    In a real system, this would be more sophisticated or user-defined.
    """
    target_column = df.columns[-1]
    target = df[target_column]
    
    # Determine if classification or regression
    unique_count = len(target.unique())
    if unique_count < 10 or target.dtype == 'object' or target.dtype == 'bool' or target.dtype == 'category':
        problem_type = 'classification'
        
        # Convert string/object targets to integers
        if target.dtype == 'object' or target.dtype == 'category':
            target_map = {val: i for i, val in enumerate(target.unique())}
            target = target.map(target_map)
    else:
        problem_type = 'regression'
    
    return target, target_column, problem_type

def encode_categorical_features(df, cat_columns, encoders=None):
    """
    Encode categorical features using one-hot encoding.
    
    Args:
        df (pd.DataFrame): The dataframe to encode
        cat_columns (list): List of categorical columns
        encoders (dict): Dictionary of encoders from training phase
        
    Returns:
        tuple: (encoded_df, encoders_dict) for training or encoded_df for prediction
    """
    encoded_df = df.copy()
    
    if encoders is None:
        # Training phase - create new encoders
        encoders = {}
        
        for col in cat_columns:
            # Store category mapping
            encoders[col] = {
                'categories': df[col].unique().tolist(),
                'mapping': {}
            }
            
            # Create dummy variables for each category
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            
            # Store mapping for each column
            for dummy_col in dummies.columns:
                encoders[col]['mapping'][dummy_col] = True
            
            # Add dummy columns to dataframe
            encoded_df = pd.concat([encoded_df, dummies], axis=1)
            
            # Drop original categorical column
            encoded_df = encoded_df.drop(col, axis=1)
        
        return encoded_df, encoders
    else:
        # Prediction phase - use existing encoders
        for col in cat_columns:
            if col in encoders:
                # Get unique categories from training
                known_categories = encoders[col]['categories']
                
                # Create dummy variables using the same categories as in training
                for cat in known_categories:
                    dummy_col = f"{col}_{cat}"
                    encoded_df[dummy_col] = (encoded_df[col] == cat).astype(int)
                
                # Drop original categorical column
                encoded_df = encoded_df.drop(col, axis=1)
            else:
                # If we encounter a new categorical column not seen during training,
                # we can either drop it or encode it on the fly
                encoded_df = encoded_df.drop(col, axis=1)
        
        return encoded_df

def normalize_numerical_features(df, num_columns):
    """Normalize numerical features to have mean=0 and std=1."""
    for col in num_columns:
        if col in df.columns:  # Check if column exists (might have been dropped)
            mean = df[col].mean()
            std = df[col].std()
            
            # Avoid division by zero
            if std == 0:
                df[col] = 0
            else:
                df[col] = (df[col] - mean) / std
    
    return df