import numpy as np
from .base_model import BaseModel

class LinearRegression(BaseModel):
    """Linear Regression implemented from scratch using only NumPy."""
    
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X, y):
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.set_feature_names(X.columns.tolist())
            
        # Convert to numpy arrays if they aren't already
        X = np.array(X)
        y = np.array(y)
        
        # Get dimensions
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.num_iterations):
            # Linear model
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute cost for history
            cost = (1/(2*n_samples)) * np.sum((y_pred - y)**2)
            self.cost_history.append(cost)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise Exception("Model is not fitted yet.")
        
        X = np.array(X)
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    
    def get_metrics(self):
        """Return model metrics as a dictionary."""
        return {
            'final_cost': self.cost_history[-1] if self.cost_history else None,
            'iterations': len(self.cost_history)
        }
    
    def get_feature_importance(self):
        """Return feature importance based on the absolute values of weights."""
        if not self.is_fitted:
            raise Exception("Model is not fitted yet.")
        
        # Use absolute weights as importance
        abs_weights = np.abs(self.weights)
        
        # Create a dictionary mapping feature names to importance
        if self.feature_names:
            raw_importances = {name: float(importance) for name, importance in 
                             zip(self.feature_names, abs_weights)}
        else:
            raw_importances = {f"feature_{i}": float(importance) for i, importance in 
                             enumerate(abs_weights)}
        
        # Use the base class method to normalize the importances
        return self.normalize_feature_importance(raw_importances)