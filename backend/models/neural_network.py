import numpy as np
from .base_model import BaseModel

class Layer:
    def __init__(self, n_inputs, n_neurons, activation='relu'):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation
        self.last_activation = None
        self.error = None
        self.delta = None

    def activate(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def activate_derivative(self, x):
        if self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation == 'sigmoid':
            s = self.activate(x)
            return s * (1 - s)
        else:
            raise ValueError(f"Unsupported activation derivative: {self.activation}")

class NeuralNetwork(BaseModel):
    """Simple Neural Network implemented from scratch using only NumPy."""
    
    def __init__(self, hidden_layer_sizes=(16, 8), learning_rate=0.01, num_iterations=1000):
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.layers = []
        self.cost_history = []
        self.problem_type = None  # 'classification' or 'regression'
        
    def _build_network(self, X, y):
        n_samples, n_features = X.shape
        
        # Determine problem type
        unique_values = np.unique(y)
        if len(unique_values) <= 10:  # Arbitrary threshold
            self.problem_type = 'classification'
            output_neurons = len(unique_values) if len(unique_values) > 2 else 1
            output_activation = 'sigmoid'
        else:
            self.problem_type = 'regression'
            output_neurons = 1
            output_activation = 'linear'
            
        # Input layer to first hidden layer
        self.layers.append(Layer(n_features, self.hidden_layer_sizes[0], 'relu'))
        
        # Hidden layers
        for i in range(1, len(self.hidden_layer_sizes)):
            self.layers.append(
                Layer(self.hidden_layer_sizes[i-1], self.hidden_layer_sizes[i], 'relu')
            )
            
        # Output layer
        self.layers.append(
            Layer(self.hidden_layer_sizes[-1], output_neurons, output_activation)
        )
    
    def _forward(self, X):
        layer_outputs = [X]
        layer_inputs = [X]
        
        for layer in self.layers:
            # Compute input to the current layer
            layer_input = np.dot(layer_outputs[-1], layer.weights) + layer.biases
            layer_inputs.append(layer_input)
            
            # Apply activation function
            layer_output = layer.activate(layer_input)
            layer_outputs.append(layer_output)
            
        return layer_outputs, layer_inputs
    
    def _compute_cost(self, y_true, y_pred):
        if self.problem_type == 'classification':
            # Binary cross-entropy
            epsilon = 1e-15  # Small value to avoid log(0)
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # Mean squared error for regression
            cost = np.mean(np.square(y_true - y_pred))
        return cost
    
    def _backward(self, X, y, layer_outputs, layer_inputs):
        # Initialize gradients
        n_samples = X.shape[0]
        
        # Compute output layer error
        output_layer = self.layers[-1]
        output = layer_outputs[-1]
        
        if self.problem_type == 'classification':
            # Binary cross-entropy derivative
            output_error = output - y
        else:
            # MSE derivative
            output_error = 2 * (output - y) / n_samples
            
        # Backpropagate the error
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            layer_input = layer_inputs[i+1]
            prev_layer_output = layer_outputs[i]
            
            if i == len(self.layers) - 1:  # Output layer
                layer.error = output_error
            else:  # Hidden layers
                next_layer = self.layers[i+1]
                layer.error = np.dot(next_layer.delta, next_layer.weights.T)
                
            # Compute delta (derivative of cost with respect to layer input)
            layer.delta = layer.error * layer.activate_derivative(layer_input)
            
            # Compute gradients
            layer.dweights = np.dot(prev_layer_output.T, layer.delta) / n_samples
            layer.dbiases = np.sum(layer.delta, axis=0, keepdims=True) / n_samples
    
    def _update_params(self):
        for layer in self.layers:
            layer.weights -= self.learning_rate * layer.dweights
            layer.biases -= self.learning_rate * layer.dbiases
    
    def fit(self, X, y):
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.set_feature_names(X.columns.tolist())
            
        # Convert to numpy arrays if they aren't already
        X = np.array(X)
        y = np.array(y)
        
        # Reshape y if needed
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        # Build the network architecture
        self._build_network(X, y)
        
        # Gradient descent
        for i in range(self.num_iterations):
            # Forward pass
            layer_outputs, layer_inputs = self._forward(X)
            
            # Compute cost
            cost = self._compute_cost(y, layer_outputs[-1])
            self.cost_history.append(cost)
            
            # Backward pass
            self._backward(X, y, layer_outputs, layer_inputs)
            
            # Update parameters
            self._update_params()
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise Exception("Model is not fitted yet.")
        
        X = np.array(X)
        layer_outputs, _ = self._forward(X)
        predictions = layer_outputs[-1]
        
        # For binary classification, convert probabilities to classes
        if self.problem_type == 'classification':
            if predictions.shape[1] == 1:  # Binary classification
                return (predictions >= 0.5).astype(int)
            else:  # Multi-class classification
                return np.argmax(predictions, axis=1)
        
        return predictions
    
    def get_metrics(self):
        """Return model metrics as a dictionary."""
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'final_cost': self.cost_history[-1] if self.cost_history else None,
            'iterations': len(self.cost_history),
            'problem_type': self.problem_type
        }
    
    def get_feature_importance(self):
        """
        Estimate feature importance using first layer weights.
        While not as accurate as other methods, this provides some insight.
        """
        if not self.is_fitted:
            raise Exception("Model is not fitted yet.")
            
        # Use absolute sum of weights from input to first hidden layer as importance
        importances = np.sum(np.abs(self.layers[0].weights), axis=1)
        total = np.sum(importances)
        
        if total == 0:
            importances = np.ones_like(importances) / len(importances)
        else:
            importances = importances / total
            
        # Create a dictionary mapping feature names to importance
        if self.feature_names:
            return {name: float(importance) for name, importance in zip(self.feature_names, importances)}
        else:
            return {f"feature_{i}": float(importance) for i, importance in enumerate(importances)}