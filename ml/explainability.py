import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# Suppress warnings from SHAP/numba if any
import warnings
warnings.filterwarnings('ignore')

_pipeline = None
_X_train = None

def load_ml_assets():
    global _pipeline, _X_train
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if _pipeline is None:
        pipeline_path = os.path.join(base_dir, 'ml', 'saved_models', 'diabetes_pipeline.joblib')
        if os.path.exists(pipeline_path):
            _pipeline = joblib.load(pipeline_path)
            
    if _X_train is None:
        test_data_path = os.path.join(base_dir, 'ml', 'saved_models', 'test_data.joblib')
        if os.path.exists(test_data_path):
            _X_train = joblib.load(test_data_path)
            
    return _pipeline, _X_train

def explain_prediction_shap(features_dict):
    """
    Generate SHAP explainability values for a single patient prediction.
    
    Returns:
        dict: A dictionary containing:
            - 'explanations': list of dicts with keys 'feature', 'value', 'shap_value', 'effect'
            - 'success': bool indicating if SHAP ran successfully
            - 'explainer_type': str name of explainer used
            - 'error_msg': str (if failed)
    """
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    # Pre-build a fallback explanation in case SHAP fails or metrics are missing
    fallback_result = {
        'explanations': [],
        'success': False,
        'explainer_type': 'Fallback Heuristic',
        'error_msg': None
    }
    
    try:
        pipeline, X_train_raw = load_ml_assets()
        if pipeline is None or X_train_raw is None:
            raise FileNotFoundError("Model assets or training background data not found. Please train the model.")
            
        import shap
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        
        # 1. Clean input dict
        cleaned_dict = {}
        for col in feature_cols:
            val = features_dict.get(col, 0)
            if col in ['Pregnancies', 'Age']:
                cleaned_dict[col] = int(val)
            else:
                cleaned_dict[col] = float(val)
                
        # 2. Build single-row DataFrame
        input_df = pd.DataFrame([cleaned_dict], columns=feature_cols)
        
        # Impute zeros in input
        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in cols_with_zeros:
            if input_df.loc[0, col] == 0:
                input_df.loc[0, col] = np.nan
                
        # 3. Preprocess both baseline background data and our input data
        # SHAP explainers need to run on the inputs that are fed directly into the estimator (i.e. scaled and imputed data)
        preprocessor = Pipeline(pipeline.steps[:-1])
        classifier = pipeline.steps[-1][1]
        
        X_train_preprocessed = preprocessor.transform(X_train_raw)
        X_patient_preprocessed = preprocessor.transform(input_df)
        
        shap_values = None
        explainer_type = ""
        
        # 4. Fit Explainer based on model class
        if isinstance(classifier, (RandomForestClassifier, DecisionTreeClassifier)):
            explainer_type = "TreeExplainer"
            explainer = shap.TreeExplainer(classifier)
            raw_shap = explainer.shap_values(X_patient_preprocessed)
            
            # TreeExplainer outputs shape [n_samples, n_features, n_classes] or list of length n_classes.
            # We want metrics for class 1 (Diabetic).
            if isinstance(raw_shap, list):
                shap_values = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0]
            elif isinstance(raw_shap, np.ndarray):
                if len(raw_shap.shape) == 3: # (samples, features, classes)
                    shap_values = raw_shap[0, :, 1]
                elif len(raw_shap.shape) == 2: # (samples, features)
                    shap_values = raw_shap[0]
            else:
                shap_values = raw_shap
                
        elif isinstance(classifier, LogisticRegression):
            explainer_type = "LinearExplainer"
            explainer = shap.LinearExplainer(classifier, X_train_preprocessed)
            raw_shap = explainer.shap_values(X_patient_preprocessed)
            
            if isinstance(raw_shap, list):
                shap_values = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0]
            elif isinstance(raw_shap, np.ndarray):
                if len(raw_shap.shape) == 2: # (samples, features)
                    shap_values = raw_shap[0]
                else:
                    shap_values = raw_shap
            else:
                shap_values = raw_shap
        else:
            explainer_type = "KernelExplainer"
            # KernelExplainer fallback for generic models
            explainer = shap.KernelExplainer(classifier.predict_proba, X_train_preprocessed)
            raw_shap = explainer.shap_values(X_patient_preprocessed)
            if isinstance(raw_shap, list):
                shap_values = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0]
            elif isinstance(raw_shap, np.ndarray) and len(raw_shap.shape) == 3:
                shap_values = raw_shap[0, :, 1]
            else:
                shap_values = raw_shap[0] if hasattr(raw_shap, '__len__') else raw_shap

        # Verify we got 8 feature values
        if shap_values is not None and len(shap_values) == len(feature_cols):
            explanations = []
            for i, col in enumerate(feature_cols):
                val = cleaned_dict[col]
                # Is missing?
                is_missing = val == 0 and col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
                display_val = "Not Provided" if is_missing else f"{val}"
                
                sh_val = float(shap_values[i])
                effect = "positive" if sh_val > 0 else ("negative" if sh_val < 0 else "neutral")
                
                explanations.append({
                    'feature': col,
                    'value': display_val,
                    'shap_value': sh_val,
                    'effect': effect
                })
                
            # Sort by absolute SHAP values (highest impact first)
            explanations.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            
            return {
                'explanations': explanations,
                'success': True,
                'explainer_type': explainer_type,
                'error_msg': None
            }
        else:
            raise ValueError("Calculated SHAP values did not align with feature dimensions.")
            
    except Exception as e:
        fallback_result['error_msg'] = str(e)
        print(f"SHAP explanation failed: {str(e)}. Triggering clinical heuristic fallback.")
        
        # Clinical Heuristic Fallback
        # Generate explanations based on deviation from median and general weights
        try:
            # Load stats from training if we can
            # If not, use standard reference medians
            reference_medians = {
                'Pregnancies': 3.0,
                'Glucose': 117.0,
                'BloodPressure': 72.0,
                'SkinThickness': 23.0,
                'Insulin': 30.5,
                'BMI': 32.0,
                'DiabetesPedigreeFunction': 0.3725,
                'Age': 29.0
            }
            
            feature_weights = {
                'Glucose': 0.35,
                'BMI': 0.20,
                'Age': 0.15,
                'DiabetesPedigreeFunction': 0.12,
                'Pregnancies': 0.08,
                'BloodPressure': 0.05,
                'SkinThickness': 0.03,
                'Insulin': 0.02
            }
            
            explanations = []
            for col in feature_cols:
                val = float(features_dict.get(col, 0))
                is_missing = val == 0 and col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
                display_val = "Not Provided" if is_missing else f"{val}"
                
                if is_missing:
                    sh_val = 0.0
                else:
                    med = reference_medians[col]
                    # Compute simple relative deviation
                    if med > 0:
                        deviation = (val - med) / med
                    else:
                        deviation = 0.0
                    sh_val = deviation * feature_weights[col]
                    
                effect = "positive" if sh_val > 0.01 else ("negative" if sh_val < -0.01 else "neutral")
                
                explanations.append({
                    'feature': col,
                    'value': display_val,
                    'shap_value': sh_val,
                    'effect': effect
                })
                
            explanations.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            fallback_result['explanations'] = explanations
            return fallback_result
        except Exception as fallback_err:
            fallback_result['error_msg'] = f"SHAP failed: {str(e)}. Fallback failed: {str(fallback_err)}"
            return fallback_result
