import os
import numpy as np
import tensorflow as tf
import pickle
import scipy.io # Added for .mat support
from flask import Flask, request, jsonify, render_template

# Initialize Flask App
app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'best_model.keras'
SCALER_PATH = 'scaler.pkl'
OPTIMAL_THRESHOLD = 0.4195

# --- STATIC MODEL METRICS (From your Training Phase) ---
# These are the results you achieved during training/testing
MODEL_PERFORMANCE = {
    "accuracy": "72.5%",
    "precision": "68.2%",
    "auc": "0.648",
    "f1_score": "0.65"
}

# --- LOAD MODEL & SCALER ---
print("Loading model and scaler...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Model and Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    print("Make sure best_model.keras and scaler.pkl are in the same folder.")

def load_fnc_data(file_obj, filename):
    """Helper to extract FNC data from either .npy or .mat"""
    try:
        if filename.endswith('.npy'):
            data = np.load(file_obj)
        elif filename.endswith('.mat'):

            file_content = file_obj.read()

            from io import BytesIO
            with BytesIO(file_content) as f:
                mat_data = scipy.io.loadmat(f)

            data = None
            for key in mat_data:

                if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                    if mat_data[key].size > 100:
                        data = mat_data[key]
                        break

            if data is None:
                print(f"Could not find valid FNC data variable in {filename}")
                return None

        else:
            return None

        if data.ndim > 1:
            data = data.flatten()
        return data

    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None

def load_icn_data(file_obj, filename):
    """Load ICN time course data (T, 105) - Optional for extended analysis"""
    try:
        if filename.endswith('.npy'):
            data = np.load(file_obj)
        elif filename.endswith('.mat'):
            file_content = file_obj.read()
            from io import BytesIO
            with BytesIO(file_content) as f:
                mat_data = scipy.io.loadmat(f)

            # Look for time course matrix with shape (T, 105)
            for key in mat_data:
                if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                    if mat_data[key].ndim == 2 and mat_data[key].shape[1] == 105:
                        print(f"✓ Found ICN time course data in {filename}: shape {mat_data[key].shape}")
                        return mat_data[key]
        else:
            return None

        # Check if loaded numpy array has correct shape
        if data is not None and data.ndim == 2 and data.shape[1] == 105:
            print(f"✓ Found ICN time course data in {filename}: shape {data.shape}")
            return data

        return None
    except Exception as e:
        print(f"Error reading ICN file {filename}: {e}")
        return None

@app.route('/')
def home():
    # Pass performance metrics to the dashboard on load
    return render_template('index.html', metrics=MODEL_PERFORMANCE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'})

    files = request.files.getlist('files')
    target_data = None
    icn_data = None
    found_filename = ""
    icn_filename = ""

    # --- FOLDER PARSING LOGIC ---
    # Loop through all files in the uploaded folder to find FNC and ICN files
    for file in files:
        if 'fnc.npy' in file.filename or 'fnc.mat' in file.filename:
            target_data = load_fnc_data(file, file.filename)
            found_filename = file.filename
        # Look for ICN time courses (optional)
        elif 'icn' in file.filename.lower() or 'timecourse' in file.filename.lower() or 'tc' in file.filename.lower():
            icn_data = load_icn_data(file, file.filename)
            if icn_data is not None:
                icn_filename = file.filename

    if target_data is None:
        return jsonify({'error': 'Could not find "fnc.npy" or "fnc.mat" in the uploaded folder.'})

    try:
        # Validate Shape
        if target_data.shape[0] != 5460:
             return jsonify({'error': f'Invalid Data Shape in {found_filename}. Expected 5460 features, got {target_data.shape[0]}'})

        # Preprocess
        data_reshaped = target_data.reshape(1, -1)
        data_scaled = scaler.transform(data_reshaped)

        # Predict
        prediction_prob = model.predict(data_scaled, verbose=0)[0][0]

        # Apply Threshold Logic
        if prediction_prob >= OPTIMAL_THRESHOLD:
            result = "Bipolar Disorder (BP)"
            confidence = prediction_prob * 100
            predicted_class = 1
        else:
            result = "Schizophrenia (SZ)"
            confidence = (1.0 - prediction_prob) * 100
            predicted_class = 0

        response_data = {
            'result': result,
            'confidence': f"{confidence:.2f}%",
            'raw_score': float(prediction_prob),
            'filename': found_filename,
            'icn_detected': icn_data is not None,
            'icn_filename': icn_filename if icn_data is not None else None,
            'icn_shape': list(icn_data.shape) if icn_data is not None else None
        }

        # Verification Logic
        ground_truth = request.form.get('ground_truth')
        if ground_truth and ground_truth != '':
            ground_truth_class = int(ground_truth)
            response_data['is_correct'] = (predicted_class == ground_truth_class)

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': f"Processing Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
