# DIY AutoML Platform

![DIY AutoML Banner](https://via.placeholder.com/800x200?text=DIY+AutoML+Platform)

## Table of Contents

1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [Key Features](#key-features)
4. [System Architecture](#system-architecture)
5. [Technical Implementation](#technical-implementation)
   - [Data Processing Pipeline](#data-processing-pipeline)
   - [Feature Engineering](#feature-engineering)
   - [Model Selection Logic](#model-selection-logic)
   - [Machine Learning Models](#machine-learning-models)
   - [Evaluation Metrics](#evaluation-metrics)
6. [Frontend Implementation](#frontend-implementation)
7. [Backend Implementation](#backend-implementation)
8. [API Reference](#api-reference)
9. [Installation](#installation)
10. [Usage Guide](#usage-guide)
11. [Examples](#examples)
12. [Future Improvements](#future-improvements)
13. [Contributing](#contributing)
14. [License](#license)

---

## Introduction

DIY AutoML is an open-source Automated Machine Learning platform that democratizes the power of machine learning by allowing users to train and deploy models without writing code. Unlike existing solutions that rely on external libraries or services, DIY AutoML implements machine learning algorithms from scratch using only NumPy, making it both educational and transparent.

The platform automatically handles data preprocessing, feature engineering, model selection, training, and evaluation - all through an intuitive web interface. Users simply upload their data, and the system handles the rest, providing detailed insights into the modeling process.

---

## Motivation

### The Problem

Machine learning has become increasingly important across industries, but several barriers prevent its widespread adoption:

1. **Technical expertise requirements**: Most ML frameworks require coding skills and statistical knowledge.
2. **Black-box models**: Many AutoML solutions hide implementation details, limiting educational value.
3. **Dependency on third-party libraries**: This creates issues with version compatibility and deployment.
4. **High computational requirements**: Many solutions require significant computing resources.

### Our Solution

DIY AutoML addresses these challenges by:

1. **No-code interface**: Users only need to upload CSV files to train models.
2. **Transparent implementations**: All algorithms are implemented from scratch with NumPy.
3. **Educational value**: The platform explains its decisions and calculations.
4. **Lightweight design**: Models are optimized for performance on standard hardware.

---

## Key Features

- **Automatic preprocessing**: Handles missing values, categorical encoding, and feature scaling.
- **Smart model selection**: Chooses the best algorithm based on data characteristics.
- **Custom model implementations**: Linear regression, logistic regression, decision trees, and more built from scratch.
- **Feature engineering**: Automatically creates new features to improve model performance.
- **Interactive visualizations**: See feature importance, model performance, and more.
- **Model download**: Export trained models for use in other applications.
- **Prediction downloads**: Get predictions as CSV files.
- **Problem type detection**: Automatically classifies regression vs. classification problems.

---

## System Architecture

DIY AutoML follows a client-server architecture:

```
┌───────────────┐     HTTP/REST     ┌───────────────┐
│               │<----------------->│               │
│   Frontend    │                   │    Backend    │
│  (Next.js)    │                   │    (Flask)    │
│               │                   │               │
└───────────────┘                   └───────────────┘
                                           │
                                           │
                                           ▼
                                    ┌───────────────┐
                                    │  Custom ML    │
                                    │ Implementations│
                                    │   (NumPy)     │
                                    │               │
                                    └───────────────┘
```

### Frontend (Next.js + React)

- User interface for data upload and result visualization.
- Interactive components for model insights.
- Result visualization and download capabilities.

### Backend (Flask)

- RESTful API for data processing.
- File handling and session management.
- Machine learning pipeline orchestration.

### Machine Learning Core (NumPy)

- Custom implementations of ML algorithms.
- Data preprocessing utilities.
- Model selection logic.
- Evaluation metrics calculation.

---

## Technical Implementation

### Data Processing Pipeline

The data processing pipeline consists of several sequential steps:

1. **Data Loading**: CSV files are parsed into pandas DataFrames.
2. **Target Identification**: The system automatically identifies the target variable (typically the last column).
3. **Problem Type Detection**: Classification vs. regression is determined based on the target variable's characteristics.
4. **Missing Value Handling**: Numerical values are imputed with median, categorical with mode.
5. **Categorical Encoding**: Non-numerical features are one-hot encoded.
6. **Feature Scaling**: Numerical features are normalized to have zero mean and unit variance.

**Code Example: Data Preprocessing**

```python
def preprocess_data(data, is_training=True, target_column=None, encoders=None):
    """
    Preprocess the data for machine learning
    - Identify numerical and categorical features
    - Handle missing values
    - Encode categorical variables
    - Normalize numerical features
    - Identify target column in training data
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
  
    # Handle missing values
    df = handle_missing_values(df)
  
    # Select target column for training data
    if is_training:
        y, target_column, problem_type = select_target_column(df)
        X = df.drop(target_column, axis=1)
        encoders = {}  # Initialize encoders dictionary
    else:
        if target_column is not None and target_column in df.columns:
            X = df.drop(target_column, axis=1)
        else:
            X = df
  
    # Identify numerical and categorical features
    cat_columns = X.select_dtypes(include=['object', 'category']).columns
    num_columns = X.select_dtypes(include=np.number).columns
  
    # Handle categorical features
    if is_training:
        X, encoders = encode_categorical_features(X, cat_columns, None)
    else:
        X = encode_categorical_features(X, cat_columns, encoders)
  
    # Normalize numerical features
    X = normalize_numerical_features(X, num_columns)
  
    if is_training:
        encoders['column_order'] = list(X.columns)
        return X, y, target_column, problem_type, encoders
    else:
        return X

def handle_missing_values(df):
    """Fill missing values in the dataframe."""
    # For numerical columns, fill with median
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # For categorical columns, fill with mode
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
  
    return df
```

### Feature Engineering

Feature engineering creates new features to improve model performance:

1. **Polynomial Features**: For regression problems, the system can create polynomial features to capture nonlinear relationships.
2. **Interaction Terms**: Products of features to capture relationships between variables.
3. **Aggregation Features**: Statistical aggregates for grouped data.
4. **Derived Features**: Mathematical transformations like log, square root, etc.

**Code Example: Feature Engineering**

```python
def engineer_features(X_train, X_test, problem_type):
    """Apply feature engineering to the training and test data."""
    # Ensure both dataframes have the same columns
    combined_columns = set(X_train.columns) | set(X_test.columns)
    for col in combined_columns:
        if col not in X_train.columns:
            X_train[col] = 0
        if col not in X_test.columns:
            X_test[col] = 0
  
    # Reorder columns to match
    X_test = X_test[X_train.columns]
  
    # Create polynomial features for regression problems
    if problem_type == 'regression' and X_train.shape[1] <= 10:  # Only for smaller feature sets
        X_train, X_test = add_polynomial_features(X_train, X_test, degree=2)
  
    # Create interaction terms between important features
    X_train, X_test = add_interaction_terms(X_train, X_test)
  
    # Add sum and mean of numerical features as new features
    X_train, X_test = add_aggregate_features(X_train, X_test)
  
    return X_train, X_test

def add_polynomial_features(X_train, X_test, degree=2):
    """Add polynomial features of specified degree."""
    # Only use top numerical features to avoid explosion in feature count
    num_cols = X_train.select_dtypes(include=np.number).columns
    if len(num_cols) > 5:  # Limit to top 5 numerical features
        # Use variance as a simple heuristic for importance
        variances = X_train[num_cols].var()
        top_cols = variances.nlargest(5).index
    else:
        top_cols = num_cols
  
    # Create polynomial features
    for col in top_cols:
        for d in range(2, degree + 1):
            X_train[f"{col}^{d}"] = X_train[col] ** d
            X_test[f"{col}^{d}"] = X_test[col] ** d
  
    return X_train, X_test
```

### Model Selection Logic

The model selection algorithm analyzes dataset characteristics to determine the most appropriate model:

1. **Dataset size**: Small vs. large datasets require different approaches.
2. **Feature count**: The number of features influences model choice.
3. **Problem type**: Classification vs. regression.
4. **Feature correlations**: Linear vs. nonlinear relationships.
5. **Class distribution**: For classification, balanced vs. imbalanced classes.

**Code Example: Model Selection**

```python
def select_model_type(X, y, problem_type):
    """Select appropriate model based on data characteristics."""
    n_samples, n_features = X.shape
  
    # Check data size
    small_data = n_samples < 1000
    large_data = n_samples > 10000
  
    # Check feature count
    few_features = n_features < 10
    many_features = n_features > 50
  
    if problem_type == 'classification':
        # Get class distribution
        class_counts = np.bincount(y.astype(int))
        n_classes = len(class_counts)
      
        # Check for class imbalance
        class_ratio = np.min(class_counts) / np.max(class_counts) if np.max(class_counts) > 0 else 0
        balanced_classes = class_ratio > 0.3
      
        # Check if binary or multiclass
        binary = n_classes <= 2
      
        # Simple decision rules for classification
        if small_data:
            if binary and few_features:
                return 'logistic_regression'
            elif few_features and balanced_classes:
                return 'decision_tree'
            else:
                return 'random_forest'
        elif large_data:
            if many_features:
                return 'random_forest'
            elif binary:
                return 'svm'
            else:
                return 'neural_network'
        else:  # Medium data size
            if binary and few_features:
                return 'logistic_regression'
            elif few_features:
                return 'svm'
            else:
                return 'random_forest'
  
    else:  # Regression
        # Check feature correlations
        if few_features:
            try:
                # Calculate correlation matrix
                corr_matrix = X.corr().abs()
                # Calculate mean correlation (excluding self-correlations)
                mean_corr = (corr_matrix.sum().sum() - n_features) / (n_features * (n_features - 1))
                linear_relationship = mean_corr > 0.3
            except:
                linear_relationship = False
        else:
            linear_relationship = False
      
        # Simple decision rules for regression
        if small_data:
            if few_features and linear_relationship:
                return 'linear_regression'
            else:
                return 'decision_tree'
        elif large_data:
            if many_features:
                return 'random_forest'
            else:
                return 'neural_network'
        else:  # Medium data size
            if few_features and linear_relationship:
                return 'linear_regression'
            else:
                return 'random_forest'
```

### Machine Learning Models

#### Linear Regression

**Mathematical Foundation**:

For features X = [x₁, x₂, ..., xₚ], the linear regression model predicts:

y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε

Where:

- y is the predicted value.
- β₀ is the intercept.
- β₁, β₂, ..., βₚ are the coefficients.
- ε is the error term.

The objective is to minimize the cost function:

J(β) = (1/2m) * Σ(yᵢ - ŷᵢ)²

**Code Example: Linear Regression Implementation**

```python
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
```

#### Logistic Regression

**Mathematical Foundation**:

The logistic regression model applies a sigmoid function to a linear combination of features:

P(y=1|X) = 1 / (1 + e^(-z))

Where:

- z = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ.
- P(y=1|X) is the probability of the positive class.

**Code Example: Logistic Regression Implementation**

```python
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
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to avoid overflow
  
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
            cost = -(1/n_samples) * np.sum(y*np.log(y_pred + 1e-15) + 
                                          (1-y)*np.log(1-y_pred + 1e-15))
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
```

### Evaluation Metrics

DIY AutoML calculates various metrics to evaluate model performance:

**Code Example: Evaluation Metrics**

```python
def calculate_regression_metrics(y_true, y_pred):
    """Calculate common regression metrics."""
    n_samples = len(y_true)
  
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y_true - y_pred))
  
    # Mean Squared Error (MSE)
    mse = np.mean((y_true - y_pred) ** 2)
  
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
  
    # R-squared
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
  
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

def calculate_classification_metrics(y_true, y_pred):
    """Calculate common classification metrics."""
    # Ensure arrays are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
  
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
  
    # For binary classification
    if len(np.unique(y_true)) == 2:
        # True positives, false positives, true negatives, false negatives
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
      
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
      
        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
      
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
      
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
  
    # For multi-class, return just accuracy
    return {'Accuracy': accuracy}
```

---

## Frontend Implementation

### Technology Stack

- **Next.js**: React framework for server-rendered applications.
- **Tailwind CSS**: Utility-first CSS framework for styling.
- **Axios**: HTTP client for API requests.
- **Recharts**: Charting library for visualizations.

### Key Components

1. **FileUpload Component**:

   - Handles file selection and drag-and-drop functionality.
   - Validates file types and provides user feedback.
2. **ModelResult Component**:

   - Displays model training results.
   - Shows feature importance, metrics, and prediction options.
   - Tabbed interface for organized information display.
3. **LoadingState Component**:

   - Animated loading indicator during model training.
   - Progress feedback for users during processing.

---

## Backend Implementation

### Technology Stack

- **Flask**: Lightweight web framework for Python.
- **NumPy**: Scientific computing library, used for ML algorithms.
- **Pandas**: Data manipulation and analysis.
- **Werkzeug**: Utilities for WSGI applications, used for file handling.

### API Endpoints

1. **Health Check**:

   - Endpoint: `GET /api/health`
   - Purpose: Verify API is running.
2. **AutoML Processing**:

   - Endpoint: `POST /api/automl`
   - Purpose: Process uploaded files and run the AutoML pipeline.
   - Input: Training and prediction CSV files.
   - Output: Model type, metrics, feature importance, download URLs.
3. **Download Predictions**:

   - Endpoint: `GET /api/download/<session_id>/predictions`
   - Purpose: Download generated predictions as CSV.
4. **Download Model**:

   - Endpoint: `GET /api/download/<session_id>/model`
   - Purpose: Download trained model as ZIP file.

---

## Installation

### Prerequisites

- Node.js (v14+).
- Python (v3.8+).
- pip.
- npm or yarn.

### Backend Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/diy-automl.git
   cd diy-automl
   ```
2. Create a Python virtual environment:

   ```bash
   cd backend
   python -m venv venv
   ```
3. Activate the virtual environment:

   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
5. Run the Flask server:

   ```bash
   python app.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:

   ```bash
   cd ../frontend
   ```
2. Install dependencies:

   ```bash
   npm install
   ```
3. Run the development server:

   ```bash
   npm run dev
   ```

Visit [http://localhost:3000](http://localhost:3000) to access the application.

---

## Usage Guide

### Step 1: Prepare Your Data

1. Create a training CSV file with:

   - Feature columns.
   - Target variable (last column).
2. Create a prediction CSV file with:

   - The same feature columns as the training data.
   - No target column.

### Step 2: Upload Files

1. Navigate to the DIY AutoML website.
2. Click "Choose file" to select your training data.
3. Click "Choose file" to select your prediction data.
4. Click "Generate Predictions".

### Step 3: Analyze Results

Once processing is complete, you'll see:

- The selected model type.
- Feature importance visualization.
- Model performance metrics.
- Options to download:
  - Predictions as CSV.
  - Trained model as a ZIP file.

---

## Future Improvements

### Short-term Roadmap

1. **Algorithm Improvements**:

   - Implement ensemble methods like XGBoost and AdaBoost.
   - Add support for deep learning with NumPy.
2. **User Experience**:

   - Add data visualization tools for exploratory analysis.
   - Implement automated report generation.
   - Add model explainability features.
3. **System Enhancements**:

   - User accounts and saved projects.
   - API for programmatic access.
   - Hyperparameter tuning options.

---

## Contributing

We welcome contributions to DIY AutoML! Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Make your changes.
4. Test thoroughly.
5. Commit your changes: `git commit -am 'Add feature-name'`.
6. Push to the branch: `git push origin feature-name`.
7. Submit a pull request.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
