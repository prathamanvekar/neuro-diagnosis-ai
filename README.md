# Neuro-Diagnosis AI

## Project Overview

Neuro-Diagnosis AI is a web-based artificial intelligence system for automated classification of neuropsychiatric disorders from brain imaging data. The application uses deep learning to distinguish between **Schizophrenia (SZ)** and **Bipolar Disorder (BP)** based on functional connectivity (FNC) data extracted from fMRI scans.

### Key Features

- 🧠 **AI-Powered Classification**: Dense Neural Network (DNN) model with optimized diagnostic threshold
- 📊 **Interactive Dashboard**: Real-time predictions with confidence scores and probability visualization
- 📁 **Multi-Format Support**: Accepts both `.npy` and `.mat` file formats for fMRI data
- 📈 **Performance Metrics**: Displays model accuracy (72.5%), AUC (0.648), precision (68.2%), and F1 score (0.65)
- ✅ **Ground Truth Verification**: Optional verification against known diagnoses for validation
- 🔍 **Flexible Data Input**: Automatic detection of FNC and ICN time-course data from subject folders

## Project Structure

```
├── app.py                 # Flask backend with ML prediction engine
├── requirements.txt       # Python dependencies
├── best_model.keras       # Pre-trained neural network model
├── best_model_vgg.keras   # Alternative VGG-based model
├── best_model_xgboost.json# XGBoost alternative model
├── scaler.pkl            # Feature scaler for data normalization
├── templates/
│   └── index.html        # Web interface
├── static/
│   └── css/
│       └── styles.css    # Dashboard styling
└── data/
    ├── train/            # Training data (Schizophrenia & Bipolar samples)
    └── test/             # Test data (unlabeled subjects)
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or download the project** to your local machine

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - Flask (web framework)
   - TensorFlow (deep learning)
   - NumPy (numerical computing)
   - scikit-learn (data preprocessing)
   - SciPy (scientific computing)

3. **Ensure model files are present** in the project root:
   - `best_model.keras` - The pre-trained model
   - `scaler.pkl` - Feature scaler

## Running the Application

1. **Start the Flask server**:

   ```bash
   python app.py
   ```

   The server will start on `http://localhost:5001`

2. **Open your web browser** and navigate to:

   ```
   http://localhost:5001
   ```

3. **Use the interface**:
   - Click "Select Subject Folder" and upload a folder containing `fnc.npy` or `fnc.mat` files
   - (Optional) Select a ground truth diagnosis for verification
   - Click "Run Diagnosis" to get predictions

## Expected Input Format

The application expects subject folders with:

- **fnc.npy** or **fnc.mat**: Functional Connectivity matrix (5460 features when flattened)
- **icn_tc.npy** or **icn_tc.mat** (optional): ICN time-course data (shape: T × 105)

Example structure:

```
Subject_Folder/
├── fnc.npy
├── icn_tc.npy
└── other_files...
```

## How It Works

1. **Data Loading**: Accepts .npy and .mat formats, automatically detecting and extracting FNC data
2. **Preprocessing**: Normalizes input features using a fitted StandardScaler
3. **Prediction**: Passes processed data through the trained DNN model
4. **Classification**: Applies an optimized threshold (0.4195) to classify as SZ or BP
5. **Confidence Scoring**: Returns prediction probability and confidence percentage
6. **Verification**: Optionally compares predictions against ground truth labels

## Model Performance

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 72.5% |
| Precision | 68.2% |
| AUC Score | 0.648 |
| F1 Score  | 0.65  |

## API Endpoints

| Endpoint   | Method | Description                                             |
| ---------- | ------ | ------------------------------------------------------- |
| `/`        | GET    | Returns the web dashboard                               |
| `/predict` | POST   | Processes uploaded subject folder and returns diagnosis |

### POST `/predict` Response Example

```json
{
  "result": "Bipolar Disorder (BP)",
  "confidence": "75.45%",
  "raw_score": 0.7545,
  "filename": "fnc.npy",
  "icn_detected": true,
  "is_correct": true
}
```

## Error Handling

The application provides clear error messages for:

- Missing FNC files in uploaded folder
- Invalid data shapes (expected 5460 features)
- Corrupted or unreadable files
- Model loading failures

## Technical Stack

- **Backend**: Flask (Python web framework)
- **ML Framework**: TensorFlow/Keras
- **Data Processing**: NumPy, SciPy, scikit-learn
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Server**: Python Built-in Flask Development Server

## License & Usage

This is a demonstration project for educational and research purposes in neuroinformatics and AI-assisted diagnosis.

## Notes

- The model is pre-trained and ready to use for predictions
- All data is processed locally on your machine
- For production deployment, consider using a production WSGI server (e.g., Gunicorn)
- The debug mode is currently enabled in `app.py`; disable for production
