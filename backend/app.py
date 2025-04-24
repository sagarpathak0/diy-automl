from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
import json
import time
import traceback
from werkzeug.utils import secure_filename

# Import our custom modules
from utils.data_processing import preprocess_data
from utils.feature_engineering import engineer_features
from utils.model_selector import select_model_type
from models import create_model

app = Flask(__name__)
CORS(app)  # Enable CORS

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running."""
    return jsonify({'status': 'ok', 'message': 'API is running'})

@app.route('/api/automl', methods=['POST'])
def process_automl():
    """
    Main endpoint for AutoML processing.
    Takes training and prediction CSV files, selects a model, 
    trains it, and generates predictions.
    """
    # Check if files are present
    if 'training_file' not in request.files or 'prediction_file' not in request.files:
        return jsonify({'error': 'Both training and prediction files are required'}), 400
    
    training_file = request.files['training_file']
    prediction_file = request.files['prediction_file']
    
    # Validate filenames
    if training_file.filename == '' or prediction_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not (training_file.filename.endswith('.csv') and prediction_file.filename.endswith('.csv')):
        return jsonify({'error': 'Only CSV files are supported'}), 400
    
    try:
        # Create a unique session ID for this upload
        session_id = f"session_{int(time.time())}"
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        training_path = os.path.join(session_folder, secure_filename(training_file.filename))
        prediction_path = os.path.join(session_folder, secure_filename(prediction_file.filename))
        
        training_file.save(training_path)
        prediction_file.save(prediction_path)
        
        # Read CSV files
        training_data = pd.read_csv(training_path)
        prediction_data = pd.read_csv(prediction_path)
        
        # Validate data
        if training_data.empty or prediction_data.empty:
            return jsonify({'error': 'Empty CSV file uploaded'}), 400
            
        if training_data.shape[0] < 10:
            return jsonify({'error': 'Training data must have at least 10 samples'}), 400
        
        # Preprocess data - now including encoders
        X_train, y_train, target_column, problem_type, encoders = preprocess_data(training_data)
        X_test = preprocess_data(prediction_data, is_training=False, target_column=target_column, encoders=encoders)
        
        # Feature engineering
        X_train, X_test = engineer_features(X_train, X_test, problem_type)
        
        # Select and create appropriate model
        model_type = select_model_type(X_train, y_train, problem_type)
        
        # Create and train the model
        model = create_model(model_type)
        model.fit(X_train, y_train)
        
        # Save the trained model
        model_info = {
            'problem_type': problem_type,
            'target_column': target_column,
            'training_samples': X_train.shape[0],
            'features': X_train.shape[1],
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        model_paths = model.save_model(os.path.join(session_folder, 'model'), model_info)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Save original prediction data for output
        prediction_output = prediction_data.copy()
        
        # Decode predictions if needed for classification problems
        if problem_type == 'classification':
            # Store mapping for decoding if available
            target = training_data[target_column]
            if target.dtype == 'object' or target.dtype == 'category':
                unique_values = training_data[target_column].unique()
                # Map numeric predictions back to original categories
                prediction_labels = [unique_values[int(p)] if int(p) < len(unique_values) else p for p in predictions]
                prediction_output['prediction'] = prediction_labels
            else:
                prediction_output['prediction'] = predictions
        else:
            prediction_output['prediction'] = predictions
        
        # Save predictions to file
        output_file = os.path.join(session_folder, 'predictions.csv')
        prediction_output.to_csv(output_file, index=False)
        
        # Get model metrics
        metrics = {}
        if hasattr(model, 'get_metrics'):
            metrics = model.get_metrics()
        
        # Get feature importance if available
        feature_importance = {}
        if hasattr(model, 'get_feature_importance'):
            feature_importance = model.get_feature_importance()
            
            # Ensure feature importance is properly normalized
            if feature_importance:
                total = sum(feature_importance.values())
                if total > 0:  # Avoid division by zero
                    feature_importance = {k: v/total for k, v in feature_importance.items()}
        
        # Return result
        return jsonify({
            'model_type': model_type,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'download_url': f'/api/download/{session_id}/predictions',
            'model_download_url': f'/api/download/{session_id}/model',
            'problem_type': problem_type
        })
        
    except Exception as e:
        print(traceback.format_exc())  # Log the full stack trace
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<session_id>/predictions', methods=['GET'])
def download_predictions(session_id):
    """Endpoint to download prediction results as CSV."""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], session_id, 'predictions.csv')
    print(f"Attempting to download file at: {file_path}")
    
    if os.path.exists(file_path):
        print(f"File exists, sending...")
        try:
            response = send_file(
                file_path, 
                mimetype='text/csv', 
                as_attachment=True, 
                download_name='predictions.csv'
            )
            # Add CORS headers to the response
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Access-Control-Allow-Methods', 'GET')
            return response
        except Exception as e:
            print(f"Error sending file: {str(e)}")
            return jsonify({'error': f'Error sending file: {str(e)}'}), 500
    else:
        print(f"File not found at {file_path}")
        # List files in the session directory to debug
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if os.path.exists(session_dir):
            files = os.listdir(session_dir)
            print(f"Files in session directory: {files}")
        else:
            print(f"Session directory does not exist: {session_dir}")
        
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/download/<session_id>/model', methods=['GET'])
def download_model(session_id):
    """Endpoint to download the trained model."""
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], session_id, 'model', 'model.pkl')
    
    if os.path.exists(model_path):
        # Create a ZIP file containing the model and model_info
        import zipfile
        import io
        
        # Create a memory file for the ZIP
        memory_file = io.BytesIO()
        
        # Create a ZIP file
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add model file
            zipf.write(model_path, arcname='model.pkl')
            
            # Add model info if it exists
            info_path = os.path.join(app.config['UPLOAD_FOLDER'], session_id, 'model', 'model_info.json')
            if os.path.exists(info_path):
                zipf.write(info_path, arcname='model_info.json')
            
            # Add a README file
            readme_content = f"""DIY AutoML Trained Model
===========================

This ZIP archive contains a trained machine learning model created with DIY AutoML.

Files:
- model.pkl: The serialized model file (use pickle.load() to load it)
- model_info.json: Metadata about the model

Example usage in Python:
```python
import pickle

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions (your data should have the same features as training data)
import pandas as pd
data = pd.read_csv('your_data.csv')
predictions = model.predict(data)
```

"""
            zipf.writestr('README.txt', readme_content)
        
        memory_file.seek(0)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='model.zip'
        )
    else:
        return jsonify({'error': 'Model file not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)