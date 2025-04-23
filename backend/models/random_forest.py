import numpy as np
from .decision_tree import DecisionTree
from .base_model import BaseModel

class RandomForest(BaseModel):
    """Random Forest implemented from scratch using only NumPy."""
    
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, max_features=None):
        super().__init__()
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # Number of features to consider for each split
        self.trees = []
        self.feature_importances_ = None
        self.problem_type = None  # 'classification' or 'regression'
    
    def fit(self, X, y):
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.set_feature_names(X.columns.tolist())
        
        # Convert to numpy arrays if they aren't already
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Determine problem type (classification or regression)
        unique_values = np.unique(y)
        if len(unique_values) <= 10:  # Arbitrary threshold
            self.problem_type = 'classification'
        else:
            self.problem_type = 'regression'
        
        # Set max_features if not specified
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))
        
        # Initialize feature importance array
        self.feature_importances_ = np.zeros(n_features)
        
        # Create and train individual trees
        for _ in range(self.n_trees):
            # Bootstrap sample (with replacement)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Create and train a decision tree
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
            
            # Add to ensemble
            self.trees.append(tree)
            
            # Accumulate feature importances
            self.feature_importances_ += tree.feature_importances_
        
        # Normalize feature importances
        self.feature_importances_ /= self.n_trees
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise Exception("Model is not fitted yet.")
        
        X = np.array(X)
        
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Combine predictions
        if self.problem_type == 'classification':
            # Use majority voting
            # Transpose to get predictions per sample
            predictions = predictions.T
            
            # For each sample, find the most common prediction
            final_predictions = np.array([
                np.bincount(pred.astype(int)).argmax() 
                for pred in predictions
            ])
            
            return final_predictions
        else:
            # Use mean for regression
            return np.mean(predictions, axis=0)
    
    def get_metrics(self):
        """Return model metrics as a dictionary."""
        return {
            'n_trees': self.n_trees,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'max_features': self.max_features,
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