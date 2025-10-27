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
    explainer = shap.Explainer(model)
    SHAP_AVAILABLE = True
except Exception:
    explainer = None
    SHAP_AVAILABLE = False

# ----------------------------
# HELPERS
# ----------------------------
def _to_dataframe(inputs: dict):
    cols = ["network_load", "relay_reputation", "budget_utilization", "recent_win_rate"]
    return pd.DataFrame([{c: float(inputs.get(c, 0.0)) for c in cols}])

def _clamp_prob(p: float) -> float:
    p = float(p)
    return max(1e-6, min(p, 0.5))

# ----------------------------
# MAIN DECISION + EXPLAINABILITY
# ----------------------------
def decide_probability(inputs: dict, telemetry: dict = None):
    """
    Predict probability using trained ML model and generate SHAP explainability plot.
    telemetry (optional): pass live metrics from TelemetryService
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
        "plot_base64": None,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Include telemetry context if provided
    if telemetry:
        explanation["telemetry"] = telemetry

    # ----------------------------
    # SHAP Explainability
    # ----------------------------
    if SHAP_AVAILABLE and explainer is not None:
        try:
            shap_values = explainer(X)
            shap_dir = Path(BASE_DIR) / "docs" / "shap_explanations"
            shap_dir.mkdir(parents=True, exist_ok=True)
            plot_path = shap_dir / f"shap_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.png"

            # Plot waterfall
            plt.figure()
            shap.plots.waterfall(shap_values[0], show=False)
            plt.title("Per-ticket SHAP Explanation")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()

            # Base64 encode image for Swagger
            with open(plot_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            # Feature importance
            shap_imp = dict(zip(X.columns, np.abs(shap_values.values[0]).tolist()))

            # Generate human-readable reasoning
            steps = []
            contributions = shap_values.values[0]
            sorted_idx = np.argsort(np.abs(contributions))[::-1]

            for idx in sorted_idx:
                feature = X.columns[idx]
                value = inputs[feature]
                contrib = contributions[idx]
                if contrib > 0:
                    effect = "increasing the win probability"
                else:
                    effect = "decreasing the win probability"

                steps.append({
                    "feature": feature,
                    "value": round(value, 3),
                    "effect": effect,
                    "contribution": float(contrib),
                    "importance": shap_imp[feature],
                })

            # Natural-language summary
            top_feats = [s["feature"] for s in steps[:2]]
            direction = "low" if prob_p_win < 0.02 else "moderate" if prob_p_win < 0.1 else "high"
            summary = f"The model predicts a {direction} win probability mainly influenced by {top_feats[0]} and {top_feats[1]}."

            # Update explanation
            explanation.update({
                "feature_importance": shap_imp,
                "steps": steps,
                "summary": summary,
                "plot_path": str(plot_path),
                "plot_base64": img_b64
            })

        except Exception as e:
            explanation["error"] = f"SHAP explainability failed: {str(e)}"

    else:
        explanation["note"] = "SHAP not available; using baseline-only prediction."

    return prob_p_win, explanation
