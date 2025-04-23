import numpy as np
from .base_model import BaseModel

class Node:
    """Decision tree node class."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Index of feature to split on
        self.threshold = threshold  # Threshold value for the feature
        self.left = left            # Left subtree (feature < threshold)
        self.right = right          # Right subtree (feature >= threshold)
        self.value = value          # Predicted value (for leaf nodes)

class DecisionTree(BaseModel):
    """Decision Tree implemented from scratch using only NumPy."""
    
    def __init__(self, max_depth=10, min_samples_split=2):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_importances_ = None
        self.n_features = None
        self.problem_type = None  # 'classification' or 'regression'
    
    def fit(self, X, y):
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.set_feature_names(X.columns.tolist())
            
        # Convert to numpy arrays if they aren't already
        X = np.array(X)
        y = np.array(y)
        
        # Determine problem type (classification or regression)
        unique_values = np.unique(y)
        if len(unique_values) <= 10:  # Arbitrary threshold
            self.problem_type = 'classification'
        else:
            self.problem_type = 'regression'
        
        # Store number of features for later use
        self.n_features = X.shape[1]
        
        # Initialize feature importance
        self.feature_importances_ = np.zeros(self.n_features)
        
        # Build the tree
        self.root = self._grow_tree(X, y)
        
        # Normalize feature importances
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ = self.feature_importances_ / np.sum(self.feature_importances_)
        
        self.is_fitted = True
        return self
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        
        # Check stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            # Create leaf node
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)
        
        # Find the best split
        feature_idx, threshold = self._best_split(X, y)
        
        # If no split improves the criterion, create a leaf node
        if feature_idx is None:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)
        
        # Update feature importance
        self.feature_importances_[feature_idx] += 1
        
        # Split the data
        left_indices = X[:, feature_idx] < threshold
        right_indices = ~left_indices
        
        # Recursively build subtrees
        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(feature=feature_idx, threshold=threshold, left=left_subtree, right=right_subtree)
    
    def _best_split(self, X, y):
        m = X.shape[0]
        if m <= 1:
            return None, None
        
        # Calculate parent impurity
        parent_impurity = self._calculate_impurity(y)
        
        # Track best split
        best_feature, best_threshold = None, None
        best_info_gain = -1
        
        # Try each feature for splitting
        for feature_idx in range(X.shape[1]):
            # Get unique values for the feature
            thresholds = np.unique(X[:, feature_idx])
            
            # Try each threshold
            for threshold in thresholds:
                # Split the data
                left_indices = X[:, feature_idx] < threshold
                right_indices = ~left_indices
                
                # Skip if split doesn't divide the data
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                
                # Calculate impurity for children
                left_impurity = self._calculate_impurity(y[left_indices])
                right_impurity = self._calculate_impurity(y[right_indices])
                
                # Calculate information gain
                left_weight = np.sum(left_indices) / m
                right_weight = np.sum(right_indices) / m
                info_gain = parent_impurity - (left_weight * left_impurity + right_weight * right_impurity)
                
                # Update best split if this one is better
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _calculate_impurity(self, y):
        if self.problem_type == 'classification':
            # Use Gini impurity for classification
            m = len(y)
            if m == 0:
                return 0
            
            # Calculate class probabilities
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / m
            
            # Gini impurity
            return 1 - np.sum(probabilities**2)
        else:
            # Use variance for regression
            return np.var(y) if len(y) > 0 else 0
    
    def _calculate_leaf_value(self, y):
        if self.problem_type == 'classification':
            # Most common class
            return np.bincount(y.astype(int)).argmax()
        else:
            # Mean value
            return np.mean(y)
    
    def predict(self, X):
        if not self.is_fitted:
            raise Exception("Model is not fitted yet.")
        
        X = np.array(X)
        
        # Make predictions for each sample
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        # If we reach a leaf node, return its value
        if node.value is not None:
            return node.value
        
        # Decide whether to go left or right
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def get_metrics(self):
        """Return model metrics as a dictionary."""
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'problem_type': self.problem_type
        }
    
    def get_feature_importance(self):
        """Return feature importance as dictionary."""
        if not self.is_fitted:
            raise Exception("Model is not fitted yet.")
        
        if self.feature_names:
            return {name: float(importance) for name, importance in 
                   zip(self.feature_names, self.feature_importances_)}
        else:
            return {f"feature_{i}": float(importance) for i, importance in 
                   enumerate(self.feature_importances_)}