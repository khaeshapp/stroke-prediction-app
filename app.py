"""
AI Stroke Risk Prediction System - Flask Backend
================================================
A machine learning-powered backend for predicting stroke risk based on patient health data.

Features:
- RESTful API endpoints
- ML-based prediction algorithm
- Supabase PostgreSQL integration for data persistence
- CORS enabled for frontend integration
- Error handling and validation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Supabase Configuration
SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_ANON_KEY = os.environ.get('SUPABASE_ANON_KEY', '')
SUPABASE_SERVICE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY', '')
model = joblib.load("stroke_model.pkl")
# Convert input to dataframe
df = pd.DataFrame([{
    "gender": patient_data.get("gender"),
    "age": patient_data.get("age"),
    "hypertension": 1 if patient_data.get("hypertension") == "Yes" else 0,
    "heart_disease": 1 if patient_data.get("heartDisease") == "Yes" else 0,
    "ever_married": patient_data.get("everMarried"),
    "work_type": patient_data.get("workType"),
    "Residence_type": patient_data.get("residenceType"),
    "avg_glucose_level": patient_data.get("avgGlucoseLevel"),
    "bmi": patient_data.get("bmi"),
    "smoking_status": patient_data.get("smokingStatus")
}])

# Label encoding
df["ever_married"] = df["ever_married"].map({
    "Yes": 1,
    "No": 0
})

df["Residence_type"] = df["Residence_type"].map({
    "Urban": 1,
    "Rural": 0
})

# One-hot encoding
df = pd.get_dummies(
    df,
    columns=[
        "gender",
        "work_type",
        "smoking_status"
    ],
    dtype=int
)

# Model training columns
expected_columns = [
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "Residence_type",
    "avg_glucose_level",
    "bmi",

    "gender_Female",
    "gender_Male",
    "gender_Other",

    "work_type_Govt_job",
    "work_type_Never_worked",
    "work_type_Private",
    "work_type_Self-employed",
    "work_type_children",

    "smoking_status_Unknown",
    "smoking_status_formerly smoked",
    "smoking_status_never smoked",
    "smoking_status_smokes"
]

# Add missing columns
for column in expected_columns:
    if column not in df.columns:
        df[column] = 0

# Reorder columns
df = df[expected_columns]

# Prediction
pred = model.predict(df)[0]

probability = model.predict_proba(df)[0][1]

# Risk interpretation
risk_level = "high" if pred == 1 else "low"

prediction = {
    "risk_level": risk_level,
    "risk_percentage": round(float(probability) * 100, 2),
    "confidence": round(float(probability) * 100, 2),
    "risk_factors": []
}
class DatabaseManager:
    """
    Database manager for storing and retrieving predictions from Supabase
    """

    @staticmethod
    async def save_prediction(patient_data: Dict, prediction: Dict) -> bool:
        """
        Save prediction to Supabase database

        Note: In production, this would use the supabase-py client
        For this implementation, we're using REST API calls
        """
        try:
            # Prepare data for insertion
            db_record = {
                'patient_data': patient_data,
                'risk_level': prediction['risk_level'],
                'risk_percentage': prediction['risk_percentage'],
                'confidence': prediction['confidence'],
                'risk_factors': prediction['risk_factors'],
            }

            # In production, use: supabase.table('stroke_predictions').insert(db_record)
            # For now, we'll simulate successful save
            return True
        except Exception as e:
            print(f"Database error: {e}")
            return False

    @staticmethod
    async def get_history(limit: int = 10) -> List[Dict]:
        """
        Retrieve prediction history from database

        Note: In production, this would use the supabase-py client
        """
        # In production: supabase.table('stroke_predictions').select('*').order('created_at', desc=True).limit(limit)
        return []


# Initialize predictor
predictor = StrokeRiskPredictor()


def validate_patient_data(data: Dict) -> Tuple[bool, str]:
    """
    Validate patient data before prediction

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not data:
        return False, "No patient data provided"

    # Required fields for meaningful prediction
    if not data.get('age'):
        return False, "Age is required"

    try:
        age = int(data.get('age', 0))
        if age < 0 or age > 120:
            return False, "Age must be between 0 and 120"
    except ValueError:
        return False, "Age must be a valid number"

    # Validate glucose level
    glucose = data.get('avgGlucoseLevel', 0)
    if glucose:
        try:
            glucose_val = float(glucose)
            if glucose_val < 0 or glucose_val > 500:
                return False, "Glucose level must be between 0 and 500 mg/dL"
        except ValueError:
            return False, "Glucose level must be a valid number"

    # Validate BMI
    bmi = data.get('bmi', 0)
    if bmi:
        try:
            bmi_val = float(bmi)
            if bmi_val < 10 or bmi_val > 60:
                return False, "BMI must be between 10 and 60"
        except ValueError:
            return False, "BMI must be a valid number"

    return True, ""


@app.route('/', methods=['GET'])
def index():
    """API root endpoint"""
    return jsonify({
        'name': 'AI Stroke Risk Prediction API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            '/predict': 'POST - Predict stroke risk from patient data',
            '/history': 'GET - Get prediction history',
            '/health': 'GET - API health check'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_version': predictor.model_version,
        'model_accuracy': predictor.model_accuracy
    })


@app.route( methods=['POST'])
def predict():
    """
    Predict stroke risk from patient data

    Expected JSON body:
    {
        "patientData": {
            "age": number,
            "gender": string,
            "hypertension": "Yes" | "No",
            "heartDisease": "Yes" | "No",
            "everMarried": "Yes" | "No",
            "residenceType": "Urban" | "Rural",
            "avgGlucoseLevel": number,
            "bmi": number,
            "workType": string,
            "smokingStatus": string
        }
    }

    Returns:
    {
        "success": true,
        "prediction": {
            "risk_level": "low" | "medium" | "high",
            "risk_percentage": number,
            "confidence": number,
            "risk_factors": [string]
        },
        "timestamp": string
    }
    """
    try:
        # Get request data
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        # Extract patient data
        patient_data = data.get('patientData', data)

        # Validate patient data
        is_valid, error_msg = validate_patient_data(patient_data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400

        # Save to database (async, don't wait for it)
        # In production: await DatabaseManager.save_prediction(patient_data, prediction)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/history', methods=['GET'])
def get_history():
    """
    Get prediction history

    Query parameters:
        limit: number of records to return (default: 10)

    Returns:
    {
        "success": true,
        "history": [
            {
                "id": string,
                "patient_data": {...},
                "risk_level": string,
                "risk_percentage": number,
                "confidence": number,
                "risk_factors": [string],
                "created_at": string
            }
        ]
    }
    """
    try:
        limit = request.args.get('limit', 10, type=int)
        limit = min(limit, 50)  # Cap at 50 records

        # In production: history = await DatabaseManager.get_history(limit)
        # For demo, return empty list
        history = []

        return jsonify({
            'success': True,
            'history': history,
            'count': len(history)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to fetch history: {str(e)}'
        }), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the prediction model"""
    return jsonify({
        'model_name': 'Stroke Risk Predictor',
        'version': predictor.model_version,
        'accuracy': predictor.model_accuracy,
        'weights': predictor.WEIGHTS,
        'description': 'Weighted scoring algorithm simulating ML model behavior'
    })


if __name__ == '__main__':
    # Development server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
