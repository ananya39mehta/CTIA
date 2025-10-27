# app/agent_ml.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import base64
from datetime import datetime
from pathlib import Path

# ----------------------------
# PATH SETUP
# ----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "probability_model.pkl")
INFO_PATH = os.path.join(MODELS_DIR, "model_info.json")

# ----------------------------
# MODEL LOADING
# ----------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run notebooks/eda_and_train.py first.")
    return joblib.load(MODEL_PATH)

model = load_model()

# Load model info metadata
if os.path.exists(INFO_PATH):
    with open(INFO_PATH, "r") as f:
        model_info = json.load(f)
else:
    model_info = {}

# ----------------------------
# SHAP EXPLAINER SETUP
# ----------------------------
try:
    SHAP_AVAILABLE = True
    explainer = shap.Explainer(model)
except Exception:
    SHAP_AVAILABLE = False
    explainer = None

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def _to_dataframe(inputs: dict):
    cols = ["network_load", "relay_reputation", "budget_utilization", "recent_win_rate"]
    return pd.DataFrame([{c: float(inputs.get(c, 0.0)) for c in cols}])

def _clamp_prob(p: float) -> float:
    p = float(p)
    return max(1e-6, min(p, 0.5))

# ----------------------------
# MAIN DECISION FUNCTION
# ----------------------------
def decide_probability(inputs: dict):
    """
    Predict probability using trained ML model and generate SHAP explainability plot.
    Returns (probability, explanation_dict)
    """
    X = _to_dataframe(inputs)
    prob_p_win = float(model.predict(X)[0])
    prob_p_win = _clamp_prob(prob_p_win)

    explanation = {
        "method": "ml_model",
        "inputs": inputs,
        "final_prob": prob_p_win,
        "feature_importance": {},
        "steps": [],
        "plot_path": None,
        "plot_base64": None
    }

    # ----------------------------
    # SHAP explainability (per-ticket)
    # ----------------------------
    if SHAP_AVAILABLE and explainer is not None:
        try:
            shap_values = explainer(X)

            # Save per-ticket SHAP waterfall plot
            shap_dir = Path(BASE_DIR) / "docs" / "shap_explanations"
            shap_dir.mkdir(parents=True, exist_ok=True)
            plot_path = shap_dir / f"shap_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.png"

            plt.figure()
            shap.plots.waterfall(shap_values[0], show=False)
            plt.title("Per-ticket SHAP Explanation")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()

            # Encode plot as base64
            with open(plot_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            # Extract per-feature importance
            shap_imp = dict(zip(X.columns, np.abs(shap_values.values[0]).tolist()))

            # Build detailed steps
            steps = []
            for feature, value in inputs.items():
                effect = "increase" if shap_values.values[0][list(X.columns).index(feature)] > 0 else "decrease"
                steps.append({
                    "feature": feature,
                    "value": value,
                    "effect": f"higher {feature} tends to {effect} prob",
                    "importance": shap_imp.get(feature, 0.0)
                })

            explanation.update({
                "feature_importance": shap_imp,
                "steps": steps,
                "plot_path": str(plot_path),
                "plot_base64": img_b64
            })

        except Exception as e:
            explanation["error"] = f"SHAP failed: {str(e)}"

    return prob_p_win, explanation
