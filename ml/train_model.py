import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

def train():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'diabetes.csv')
    save_dir = os.path.join(base_dir, 'ml', 'saved_models')
    os.makedirs(save_dir, exist_ok=True)

    # Delete stale model artifacts if they exist to prevent loading old weights/metrics
    stale_files = ['diabetes_pipeline.joblib', 'metrics.json', 'test_data.joblib']
    for file_name in stale_files:
        full_path = os.path.join(save_dir, file_name)
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
                print(f"Removed stale artifact: {full_path}")
            except Exception as e:
                print(f"Warning: Could not remove stale artifact {full_path}: {e}")

    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Features and target
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    target_col = 'Outcome'

    X = df[feature_cols].copy()
    y = df[target_col]

    # Replace impossible zeros with NaN
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zeros:
        X[col] = X[col].replace(0, np.nan)

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define Preprocessing Pipeline
    # Using SimpleImputer with median strategy on training data, followed by StandardScaler
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    # Define the three classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=5),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
    }

    results = {}
    pipelines = {}

    for name, clf in classifiers.items():
        print(f"Training {name}...")
        # Build individual pipeline
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
        
        # Fit model
        pipeline.fit(X_train, y_train)
        pipelines[name] = pipeline

        # Predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        # ROC Curve points
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        # Extract feature importances or coefficients
        fitted_clf = pipeline.named_steps['classifier']
        if hasattr(fitted_clf, 'feature_importances_'):
            importances = fitted_clf.feature_importances_
        elif hasattr(fitted_clf, 'coef_'):
            # Use absolute coefficient weights for logistic regression importances
            importances = np.abs(fitted_clf.coef_[0])
            # Normalize to sum up to 1 for visual equivalence
            if importances.sum() > 0:
                importances = importances / importances.sum()
        else:
            importances = np.ones(len(feature_cols)) / len(feature_cols)

        feature_importances = {feature_cols[i]: float(importances[i]) for i in range(len(feature_cols))}

        results[name] = {
            'metrics': {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc),
            },
            'confusion_matrix': cm.tolist(),
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            },
            'feature_importances': feature_importances
        }

    # Best Model Selection logic: 60% weight on ROC-AUC, 40% weight on Recall
    best_score = -1
    best_model_name = None

    for name in classifiers.keys():
        score = 0.6 * results[name]['metrics']['roc_auc'] + 0.4 * results[name]['metrics']['recall']
        print(f"{name} Combined Score (0.6*AUC + 0.4*Recall): {score:.4f}")
        if score > best_score:
            best_score = score
            best_model_name = name

    print(f"\nSelected Best Model: {best_model_name}")

    # Save Best Pipeline
    best_pipeline = pipelines[best_model_name]
    best_pipeline_path = os.path.join(save_dir, 'diabetes_pipeline.joblib')
    joblib.dump(best_pipeline, best_pipeline_path)
    print(f"Saved best pipeline to {best_pipeline_path}")

    # Save training dataset X_train for SHAP explainer baseline
    test_data_path = os.path.join(save_dir, 'test_data.joblib')
    joblib.dump(X_train, test_data_path)
    print(f"Saved background training data to {test_data_path}")

    # Save metrics JSON
    metrics_path = os.path.join(save_dir, 'metrics.json')
    model_info = {
        'best_model_name': best_model_name,
        'models': results,
        'feature_cols': feature_cols
    }
    with open(metrics_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    print(f"Saved comparative metrics to {metrics_path}")

    print("Pipeline training successfully completed.")

    # Saved model verification
    print("\nSaved model verification:")
    loaded_pipeline = joblib.load(best_pipeline_path)
    estimator = loaded_pipeline.steps[-1][1]
    estimator_name = type(estimator).__name__
    if "LogisticRegression" in estimator_name:
        print("Selected model: Logistic Regression")
    elif "DecisionTreeClassifier" in estimator_name:
        print("Selected model: Decision Tree")
    elif "RandomForestClassifier" in estimator_name:
        print("Selected model: Random Forest")
    else:
        print(f"Selected model: {estimator_name}")

if __name__ == '__main__':
    train()
