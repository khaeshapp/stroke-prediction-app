@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        patient_data = data.get('patientData', data)

        is_valid, error_msg = validate_patient_data(patient_data)
        if not is_valid:
            return jsonify({'success': False, 'error': error_msg}), 400

        # ---------------------------
        # BUILD RAW INPUT DATAFRAME
        # ---------------------------
        df = pd.DataFrame([{
            "age": patient_data.get("age"),
            "hypertension": 1 if patient_data.get("hypertension") == "Yes" else 0,
            "heart_disease": 1 if patient_data.get("heartDisease") == "Yes" else 0,
            "ever_married": 1 if patient_data.get("everMarried") == "Yes" else 0,
            "Residence_type": 1 if patient_data.get("residenceType") == "Urban" else 0,
            "avg_glucose_level": patient_data.get("avgGlucoseLevel"),
            "bmi": patient_data.get("bmi"),
            "gender": patient_data.get("gender"),
            "work_type": patient_data.get("workType"),
            "smoking_status": patient_data.get("smokingStatus")
        }])

        # ---------------------------
        # ONE HOT ENCODING (same as training)
        # ---------------------------
        df = pd.get_dummies(df, columns=["gender", "work_type", "smoking_status"], dtype=int)

        # ---------------------------
        # ALIGN WITH TRAINING FEATURES (IMPORTANT FIX)
        # ---------------------------
        df = df.reindex(columns=feature_names, fill_value=0)

        # ---------------------------
        # PREDICTION
        # ---------------------------
        pred = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        prediction = {
            "risk_level": "high" if pred == 1 else "low",
            "risk_percentage": round(float(probability) * 100, 2),
            "confidence": round(float(probability) * 100, 2),
            "risk_factors": []
        }

        return jsonify({
            'success': True,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
