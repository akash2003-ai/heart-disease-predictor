from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Locate models relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'disease_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

loaded_model = None
loaded_scaler = None

def load_models():
    global loaded_model, loaded_scaler
    if loaded_model is not None and loaded_scaler is not None:
        return
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Model files not found. Expected at:\n{MODEL_PATH}\n{SCALER_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        loaded_model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        loaded_scaler = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        load_models()
    except Exception as e:
        return (str(e), 500)

    data = request.get_json()
    if not data:
        return ("Invalid JSON payload", 400)

    # Expected keys based on the notebook's predict_disease function
    expected = ['gender','age','currentSmoker','cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose']
    for k in expected:
        if k not in data:
            return (f"Missing field: {k}", 400)

    # Encode inputs same way as the notebook
    gender_encoded = 1 if str(data['gender']).lower() == 'male' else 0
    currentSmoker_encoded = 1 if str(data['currentSmoker']).lower() == 'yes' else 0
    BPMeds_encoded = 1 if str(data['BPMeds']).lower() == 'yes' else 0
    prevalentStroke_encoded = 1 if str(data['prevalentStroke']).lower() == 'yes' else 0
    prevalentHyp_encoded = 1 if str(data['prevalentHyp']).lower() == 'yes' else 0
    diabetes_encoded = 1 if str(data['diabetes']).lower() == 'yes' else 0

    # Build feature array in the same order used in the notebook
    try:
        feature = np.array([[
            gender_encoded,
            float(data['age']),
            currentSmoker_encoded,
            float(data['cigsPerDay']),
            BPMeds_encoded,
            prevalentStroke_encoded,
            prevalentHyp_encoded,
            diabetes_encoded,
            float(data['totChol']),
            float(data['sysBP']),
            float(data['diaBP']),
            float(data['BMI']),
            float(data['heartRate']),
            float(data['glucose'])
        ]])
    except Exception as e:
        return (f"Invalid numeric values: {e}", 400)

    try:
        feature_scaled = loaded_scaler.transform(feature)
        predicted_class = loaded_model.predict(feature_scaled)
        predicted_int = int(predicted_class[0])
        # Try to get probability/confidence for the positive class if model supports it
        proba = None
        try:
            if hasattr(loaded_model, 'predict_proba'):
                proba = float(loaded_model.predict_proba(feature_scaled)[0][1])
        except Exception:
            proba = None

        message = "High risk of heart disease" if predicted_int == 1 else "Low risk of heart disease"
        resp = {"message": message, "prediction": predicted_int}
        # include probability if available (rounded to 3 decimals)
        if proba is not None:
            resp['probability'] = round(proba, 3)
        else:
            resp['probability'] = None

        return jsonify(resp)
    except Exception as e:
        return (f"Prediction error: {e}", 500)


if __name__ == '__main__':
    # Run development server
    app.run(host='0.0.0.0', port=5000, debug=True)
