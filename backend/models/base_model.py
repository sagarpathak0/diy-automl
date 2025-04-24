from abc import ABC, abstractmethod
import numpy as np
import pickle
import os
import json

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
    
    def save_model(self, folder_path, model_info=None):
        """
        Save model to disk.
        
        Args:
            folder_path: Directory where model should be saved
            model_info: Additional metadata about the model
        
        Returns:
            Dictionary with paths to model files
        """
        if not self.is_fitted:
            raise Exception("Cannot save model that has not been fitted")
            
        # Create folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        
        # Save the model using pickle
        model_path = os.path.join(folder_path, 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
            
        # Save model info as JSON
        info = {
            'model_type': self.__class__.__name__,
            'feature_names': self.feature_names,
        }
        
        # Add any additional info
        if model_info:
            info.update(model_info)
            
        info_path = os.path.join(folder_path, 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
            
        # Return paths
        return {
            'model': model_path,
            'info': info_path
        }
    
    @classmethod
    def load_model(cls, model_path):
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the saved model file
        
        Returns:
            Loaded model instance
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def normalize_feature_importance(self, importances):
        """Normalize feature importance values to sum to 1.0"""
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
            
            # Normalize to sum to 1.0
            if np.sum(values) > 0:
                values = values / np.sum(values)
            else:
                values = np.ones_like(values) / len(values)
        
        # Return as dictionary
        return {name: float(importance) for name, importance in zip(keys, values)}