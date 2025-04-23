import numpy as np
from .base_model import BaseModel

class SVM(BaseModel):
    """Support Vector Machine implemented from scratch using only NumPy."""
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_iterations=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param  # Regularization parameter
        self.num_iterations = num_iterations
        self.w = None
        self.b = None
        self.cost_history = []
    
    def fit(self, X, y):
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.set_feature_names(X.columns.tolist())
            
        # Convert to numpy arrays if they aren't already
        X = np.array(X)
        
        # Transform y to be -1 or 1 for SVM
        y = np.array(y)
        y_ = np.where(y <= 0, -1, 1)
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Gradient descent
        for i in range(self.num_iterations):
            for idx, x_i in enumerate(X):
                # Calculate hinge loss derivative
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    dw = self.lambda_param * self.w
                    db = 0
                else:
                    dw = self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    db = y_[idx]
                
                # Update parameters
                self.w = self.w - self.learning_rate * dw
                self.b = self.b - self.learning_rate * db
            
            # Compute cost for history
            cost = self._compute_cost(X, y_)
            self.cost_history.append(cost)
        
        self.is_fitted = True
        return self
    
    def _compute_cost(self, X, y):
        # Calculate hinge loss
        n_samples = X.shape[0]
        distances = 1 - y * (np.dot(X, self.w) - self.b)
        distances = np.maximum(0, distances)
        hinge_loss = self.lambda_param * (np.sum(self.w ** 2)) + (1/n_samples) * np.sum(distances)
        return hinge_loss
    
    def predict(self, X):
        if not self.is_fitted:
            raise Exception("Model is not fitted yet.")
        
        X = np.array(X)
        
        # Calculate raw predictions
        raw_predictions = np.dot(X, self.w) - self.b
        
        # Return class predictions (0 or 1)
        return np.where(raw_predictions <= 0, 0, 1)
    
    def get_metrics(self):
        """Return model metrics as a dictionary."""
        return {
            'final_cost': self.cost_history[-1] if self.cost_history else None,
            'iterations': len(self.cost_history),
            'regularization': self.lambda_param
        }
    
    def get_feature_importance(self):
        """Return feature importance based on the absolute values of weights."""
        if not self.is_fitted:
            raise Exception("Model is not fitted yet.")
        
        # Normalize to get relative importance
        abs_weights = np.abs(self.w)
        total = np.sum(abs_weights)
        
        if total == 0:  # Avoid division by zero
            importances = np.ones_like(abs_weights) / len(abs_weights)
        else:
            importances = abs_weights / total
        
        # Create a dictionary mapping feature names to importance
        if self.feature_names:
            return {name: float(importance) for name, importance in zip(self.feature_names, importances)}
        else:
            return {f"feature_{i}": float(importance) for i, importance in enumerate(importances)}