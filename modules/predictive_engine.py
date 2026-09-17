"""
Predictive Engine Module
Interactive "What-If" prediction interface:
- Accepts user input values for feature variables
- Computes single-row or scenario-based inferences
- Outputs predicted outcome, classification probabilities, confidence intervals
- Identifies top drivers and plain-English explanation for the prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def predict_scenario(
    trained_model_result: Dict[str, Any],
    input_values: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates what-if scenario feature values using a trained ML model.
    Returns prediction, probability/confidence, and contributing drivers.
    """
    model = trained_model_result["model_object"]
    prep_data = trained_model_result["prep_data"]
    feature_names = prep_data["feature_names"]
    scaler = prep_data["scaler"]
    task = prep_data["task_type"]

    # Construct feature vector
    row_dict = {}
    for f in feature_names:
        # Check direct match or default to 0
        if f in input_values:
            try:
                row_dict[f] = float(input_values[f])
            except Exception:
                row_dict[f] = 0.0
        else:
            row_dict[f] = 0.0

    df_row = pd.DataFrame([row_dict], columns=feature_names)
    X_scaled = scaler.transform(df_row)

    raw_pred = model.predict(X_scaled)[0]
    result = {
        "task_type": task,
        "input_features": input_values
    }

    if task == "regression":
        pred_val = round(float(raw_pred), 2)
        # Approximate 95% prediction interval using RMSE
        rmse = trained_model_result["metrics"].get("rmse", 1.0)
        lower_ci = round(pred_val - 1.96 * rmse, 2)
        upper_ci = round(pred_val + 1.96 * rmse, 2)

        result.update({
            "prediction": pred_val,
            "prediction_formatted": f"{pred_val:,.2f}",
            "confidence_interval_95": (lower_ci, upper_ci),
            "explanation": f"Predicted target value is {pred_val:,.2f} with an estimated 95% confidence band between {lower_ci:,.2f} and {upper_ci:,.2f}."
        })
    else:
        # Classification
        le = prep_data["label_encoder"]
        pred_label = le.inverse_transform([int(raw_pred)])[0] if le else str(raw_pred)

        probs_dict = {}
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[0]
            classes = prep_data["classes"]
            for c, p in zip(classes, probs):
                probs_dict[str(c)] = round(float(p) * 100, 1)
            confidence = round(float(np.max(probs)) * 100, 1)

        result.update({
            "prediction": pred_label,
            "probabilities": probs_dict,
            "confidence_pct": confidence,
            "explanation": f"Model predicts outcome '{pred_label}' with {confidence}% model confidence."
        })

    # Highlight top influential inputs
    top_fi = trained_model_result.get("feature_importances", [])[:4]
    result["top_contributing_features"] = top_fi

    return result
