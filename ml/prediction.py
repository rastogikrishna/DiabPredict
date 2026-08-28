import os
import joblib
import pandas as pd
import numpy as np

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pipeline_path = os.path.join(base_dir, 'ml', 'saved_models', 'diabetes_pipeline.joblib')
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Trained pipeline model not found at {pipeline_path}. Please run python ml/train_model.py first.")
        _pipeline = joblib.load(pipeline_path)
    return _pipeline

def predict_diabetes_risk(features_dict):
    """
    Load the saved classifier pipeline and predict diabetes risk class, probability, and risk level category.
    
    features_dict = {
        'Pregnancies': int,
        'Glucose': float,
        'BloodPressure': float,
        'SkinThickness': float,
        'Insulin': float,
        'BMI': float,
        'DiabetesPedigreeFunction': float,
        'Age': int
    }
    
    Returns:
        predicted_class: int (0 or 1)
        probability: float (0.0 to 1.0)
        risk_level: str ("LOW", "MODERATE", "HIGH")
    """
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    # Ensure variables are converted to numeric formats
    cleaned_dict = {}
    for col in feature_cols:
        val = features_dict.get(col, 0)
        if col in ['Pregnancies', 'Age']:
            cleaned_dict[col] = int(val)
        else:
            cleaned_dict[col] = float(val)

    # Convert to DataFrame
    input_data = pd.DataFrame([cleaned_dict], columns=feature_cols)
    
    # Replace impossible zeros with NaN so the pipeline imputer can process them
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zeros:
        if input_data.loc[0, col] == 0:
            input_data.loc[0, col] = np.nan

    # Load model
    pipeline = get_pipeline()
    
    # Run prediction
    predicted_class = int(pipeline.predict(input_data)[0])
    probability = float(pipeline.predict_proba(input_data)[0][1])
    
    # Categorize Risk levels
    if probability < 0.33:
        risk_level = "LOW"
    elif probability < 0.66:
        risk_level = "MODERATE"
    else:
        risk_level = "HIGH"
        
    return {
        "prediction": predicted_class,
        "prediction_label": "Diabetes Detected" if predicted_class == 1 else "Diabetes Not Detected",
        "probability": probability,
        "non_diabetes_probability": 1.0 - probability,
        "confidence": max(probability, 1.0 - probability),
        "risk_level": risk_level
    }
