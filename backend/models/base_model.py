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