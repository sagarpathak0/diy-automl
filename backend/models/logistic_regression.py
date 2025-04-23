import numpy as np
from .base_model import BaseModel

class LogisticRegression(BaseModel):
    """Logistic Regression implemented from scratch using only NumPy."""
    
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))
    
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
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid
            y_pred = self.sigmoid(linear_model)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute cost for history (log loss)
            cost = -(1/n_samples) * np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred + 1e-15))
            self.cost_history.append(cost)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        """Return probabilities for the positive class."""
        if not self.is_fitted:
            raise Exception("Model is not fitted yet.")
        
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X):
        """Return class predictions using 0.5 as threshold."""
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
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
        
        # Normalize to get relative importance
        abs_weights = np.abs(self.weights)
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