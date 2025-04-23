from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self):
        self.is_fitted = False
        self.feature_names = None
    
    @abstractmethod
    def fit(self, X, y):
        """Fit the model to the training data."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions on new data."""
        pass
    
    def set_feature_names(self, feature_names):
        """Store feature names for later use in feature importance."""
        self.feature_names = feature_names
        
    def normalize_feature_importance(self, importances):
        """
        Normalize feature importance values to sum to 1.0
        
        Args:
            importances: Array or dictionary of raw importance values
            
        Returns:
            Dictionary mapping feature names to normalized importance values
        """
        if isinstance(importances, dict):
            values = list(importances.values())
            keys = list(importances.keys())
        else:
            values = importances
            keys = self.feature_names if self.feature_names else [f"feature_{i}" for i in range(len(values))]
        
        # Convert to numpy array for processing
        values = np.array(values)
        
        # Handle case where all importances are the same (or zero)
        if np.std(values) < 1e-10:
            # Equal distribution if all are the same
            values = np.ones_like(values) / len(values)
        else:
            # Ensure non-negative values
            values = np.maximum(0, values)
            
            # Handle case where sum is zero
            if np.sum(values) == 0:
                values = np.ones_like(values) / len(values)
            else:
                # Normalize to sum to 1.0
                values = values / np.sum(values)
        
        # Return as dictionary
        return {name: float(importance) for name, importance in zip(keys, values)}