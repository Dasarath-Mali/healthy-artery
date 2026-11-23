"""
Prediction service module for Healthy Artery.
Handles model loading, feature validation, DataFrame formatting,
and risk categorization. Completely decoupled from the GUI display.
Can also be run directly from CLI for instant terminal predictions.
"""

import os
import re
import argparse
import joblib
import pandas as pd
import numpy as np
from src.config import NUMERIC_FEATURES

# Sensible clinical boundaries based on training data distribution with margin
FEATURE_DEFINITIONS = {
    "age": {
        "display_name": "Age",
        "unit": "years",
        "min": 18.0,
        "max": 120.0,
        "default": 55.0,
        "description": "Patient age (18 - 120 years)"
    },
    "trestbps": {
        "display_name": "Resting Blood Pressure",
        "unit": "mm Hg",
        "min": 60.0,
        "max": 260.0,
        "default": 140.0,
        "description": "Resting blood pressure on hospital admission (60 - 260 mm Hg)"
    },
    "chol": {
        "display_name": "Serum Cholesterol",
        "unit": "mg/dl",
        "min": 80.0,
        "max": 650.0,
        "default": 250.0,
        "description": "Serum cholesterol in mg/dl (80 - 650 mg/dl)"
    },
    "thalach": {
        "display_name": "Max Heart Rate Achieved",
        "unit": "bpm",
        "min": 50.0,
        "max": 250.0,
        "default": 150.0,
        "description": "Maximum heart rate achieved during exercise (50 - 250 bpm)"
    },
    "oldpeak": {
        "display_name": "ST Depression (Oldpeak)",
        "unit": "mm",
        "min": 0.0,
        "max": 10.0,
        "default": 1.5,
        "description": "ST depression induced by exercise relative to rest (0.0 - 10.0 mm)"
    }
}

MODEL_REGISTRY = {
    "Logistic Regression": "models/logistic.joblib",
    "Random Forest": "models/random_forest.joblib",
    "XGBoost": "models/xgboost.joblib"
}


def load_trained_models(models_dir="models"):
    """
    Load all available trained scikit-learn pipelines into memory once at startup.
    
    Returns
    -------
    dict
        Mapping of display model name -> loaded Pipeline object.
        
    Raises
    ------
    FileNotFoundError
        If any of the required model joblib files cannot be found.
    """
    loaded_models = {}
    missing_files = []

    for display_name, rel_path in MODEL_REGISTRY.items():
        full_path = os.path.abspath(rel_path)
        if not os.path.exists(full_path):
            missing_files.append(rel_path)
        else:
            try:
                loaded_models[display_name] = joblib.load(full_path)
            except Exception as e:
                raise RuntimeError(f"Error loading model '{display_name}' from {rel_path}: {e}")

    if missing_files:
        raise FileNotFoundError(
            f"The following required model files are missing:\n"
            + "\n".join(f" - {f}" for f in missing_files)
            + "\nPlease ensure 'python -m src.train' has been run to generate them."
        )

    return loaded_models


def sanitize_input_string(raw_val):
    """
    Sanitize raw user string to extract valid numerical representation.
    Handles comma decimals, units (e.g. '140 mmhg', '55 yrs'), and whitespace.
    """
    if raw_val is None:
        return ""
    val_str = str(raw_val).strip()
    if not val_str:
        return ""
    # Replace comma with period for decimal parsing
    val_str = val_str.replace(",", ".")
    # Remove common unit suffixes if present
    val_str = re.sub(r'(?i)(mm\s*hg|mg/dl|bpm|years|yrs|mm|\s)', '', val_str)
    return val_str


def validate_input_values(raw_inputs):
    """
    Validate that user inputs are valid numeric values within sensible ranges.
    
    Parameters
    ----------
    raw_inputs : dict
        Mapping of feature_name -> raw string input from GUI or CLI.
        
    Returns
    -------
    tuple (is_valid, cleaned_values, errors)
        - is_valid : bool, True if all fields pass validation.
        - cleaned_values : dict of float values.
        - errors : dict of error strings per invalid feature.
    """
    cleaned_values = {}
    errors = {}

    for feat in NUMERIC_FEATURES:
        meta = FEATURE_DEFINITIONS[feat]
        val_str = sanitize_input_string(raw_inputs.get(feat, ""))

        if not val_str:
            errors[feat] = f"Required field. Please enter a value ({meta['min']} - {meta['max']} {meta['unit']})."
            continue

        try:
            val_float = float(val_str)
        except ValueError:
            errors[feat] = "Invalid number. Please enter numeric digits."
            continue

        if np.isnan(val_float) or np.isinf(val_float):
            errors[feat] = "Invalid numerical value."
            continue

        if val_float < meta["min"] or val_float > meta["max"]:
            errors[feat] = f"Out of range: must be between {meta['min']} and {meta['max']} {meta['unit']}."
            continue

        cleaned_values[feat] = val_float

    is_valid = len(errors) == 0
    return is_valid, cleaned_values, errors


def predict_single_instance(pipeline, cleaned_values):
    """
    Perform risk prediction on validated input values using the exact feature order
    and column names expected by the trained scikit-learn Pipeline.
    
    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The loaded pipeline with imputer, scaler (if applicable), and classifier.
    cleaned_values : dict
        Validated float values for each feature in NUMERIC_FEATURES.
        
    Returns
    -------
    dict
        - probability : float (0.0 to 1.0)
        - risk_percentage : float (0.0 to 100.0)
        - risk_category : str ("Low Risk", "Moderate Risk", "High Risk")
        - status_color : str (hex color code)
    """
    # Build single-row DataFrame with the exact feature order and types
    df_row = pd.DataFrame([cleaned_values], columns=NUMERIC_FEATURES).astype(float)

    # Obtain predicted probability for disease class (target = 1)
    proba = float(pipeline.predict_proba(df_row)[0, 1])
    risk_pct = round(proba * 100, 1)

    # Categorize risk and map to traffic-light colors
    # Green below 30%, Amber 30 to 60%, Red above 60%
    if risk_pct < 30.0:
        risk_category = "Low Risk"
        status_color = "#2ecc71"  # Clinical Green
    elif risk_pct <= 60.0:
        risk_category = "Moderate Risk"
        status_color = "#f39c12"  # Warning Amber
    else:
        risk_category = "High Risk"
        status_color = "#e74c3c"  # Critical Red

    return {
        "probability": proba,
        "risk_percentage": risk_pct,
        "risk_category": risk_category,
        "status_color": status_color,
        "input_frame": df_row
    }


def main():
    parser = argparse.ArgumentParser(description="Healthy Artery: Predict heart disease risk from CLI.")
    for feat in NUMERIC_FEATURES:
        meta = FEATURE_DEFINITIONS[feat]
        parser.add_argument(f"--{feat}", type=str, default=str(meta["default"]),
                            help=f"{meta['display_name']} ({meta['unit']}) [Default: {meta['default']}]")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="XGBoost",
                        help="Model to use for prediction (default: XGBoost)")
    args = parser.parse_args()

    models = load_trained_models()
    selected_model = models[args.model]

    input_dict = {feat: getattr(args, feat) for feat in NUMERIC_FEATURES}
    is_valid, cleaned, errors = validate_input_values(input_dict)

    if not is_valid:
        print("\n[!] Input Validation Errors:")
        for feat, err in errors.items():
            print(f"  - {feat}: {err}")
        return

    res = predict_single_instance(selected_model, cleaned)
    print("\n" + "=" * 50)
    print(f"  HEALTHY ARTERY PREDICTION [{args.model}]")
    print("=" * 50)
    print(f"Patient Profile:")
    for feat in NUMERIC_FEATURES:
        meta = FEATURE_DEFINITIONS[feat]
        print(f"  • {meta['display_name']:28s}: {cleaned[feat]} {meta['unit']}")
    print("-" * 50)
    print(f"Risk Probability : {res['probability']:.4f} ({res['risk_percentage']}%)")
    print(f"Risk Tier        : {res['risk_category']}")
    print(f"Status Color     : {res['status_color']}")
    print("=" * 50)
    print("Notice: This is an educational tool, not a medical diagnosis.\n")


if __name__ == "__main__":
    main()
