from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .decision_tree import DecisionTree
from .random_forest import RandomForest
from .svm import SVM
from .neural_network import NeuralNetwork

def create_model(model_type):
    """Factory function to create a model instance based on the specified type."""
    if model_type == 'linear_regression':
        return LinearRegression()
    elif model_type == 'logistic_regression':
        return LogisticRegression()
    elif model_type == 'decision_tree':
        return DecisionTree()
    elif model_type == 'random_forest':
        return RandomForest()
    elif model_type == 'svm':
        return SVM()
    elif model_type == 'neural_network':
        return NeuralNetwork()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")