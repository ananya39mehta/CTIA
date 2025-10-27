# app/agent_ml.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import base64
from datetime import datetime, UTC
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

if os.path.exists(INFO_PATH):
    with open(INFO_PATH, "r") as f:
        model_info = json.load(f)
else:
    model_info = {}

# ----------------------------
# SHAP EXPLAINER SETUP (robust)
# ----------------------------
def _make_explainer(model):
    """
    Creates a compatible SHAP explainer for both tree and pipeline models.
    """
    try:
        # Tree models (RandomForest, XGBoost, etc.)
        if hasattr(model, "estimators_") or "Tree" in model.__class__.__name__:
            return shap.Explainer(model)
        # Pipelines (LinearRegression, etc.)
        elif hasattr(model, "predict"):
            return shap.Explainer(model.predict, shap.maskers.Independent(np.zeros((1, 4))))
    except Exception:
        return None
    return None

explainer = _make_explainer(model)
SHAP_AVAILABLE = explainer is not None

# ----------------------------
# HELPERS
# ----------------------------
def _to_dataframe(inputs: dict):
    cols = ["network_load", "relay_reputation", "budget_utilization", "recent_win_rate"]
    return pd.DataFrame([{c: float(inputs.get(c, 0.0)) for c in cols}])

def _clamp_prob(p: float) -> float:
    return max(1e-6, min(float(p), 0.5))

# ----------------------------
# MAIN DECISION + EXPLAINABILITY
# ----------------------------
def decide_probability(inputs: dict, telemetry: dict = None):
    """
    Predict win probability using ML model + SHAP visualization.
    Produces both Waterfall & Bar plots under /docs/shap_explanations/
    and stores Base64 version for /explain_ticket_visual/.
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
        "plot_base64": None,
        "plot_path": None,
        "timestamp": datetime.now(UTC),
    }

    if telemetry:
        explanation["telemetry"] = telemetry

    shap_dir = Path(BASE_DIR) / "docs" / "shap_explanations"
    shap_dir.mkdir(parents=True, exist_ok=True)

    if SHAP_AVAILABLE and explainer is not None:
        try:
            # --- Compute SHAP values ---
            shap_values = explainer(X)
            contribs = shap_values.values[0] if hasattr(shap_values, "values") else np.zeros(len(X.columns))
            shap_imp = dict(zip(X.columns, np.abs(contribs).tolist()))

            # --- Create Waterfall Plot ---
            waterfall_path = shap_dir / f"shap_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}.png"
            plt.figure()
            shap.plots.waterfall(shap_values[0], show=False)
            plt.title("Per-Ticket SHAP Waterfall")
            plt.savefig(waterfall_path, bbox_inches="tight")
            plt.close()

            # --- Encode Base64 for Swagger display ---
            with open(waterfall_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            # --- Build feature effect list ---
            steps = []
            for i, feature in enumerate(X.columns):
                contrib = contribs[i]
                effect = "increasing" if contrib > 0 else "decreasing"
                steps.append({
                    "feature": feature,
                    "value": round(inputs[feature], 3),
                    "effect": f"{effect} the win probability",
                    "contribution": float(contrib),
                    "importance": shap_imp[feature]
                })

            # --- Summary sentence ---
            top_feats = sorted(shap_imp, key=shap_imp.get, reverse=True)[:2]
            direction = "low" if prob_p_win < 0.02 else "moderate" if prob_p_win < 0.1 else "high"
            summary = f"The model predicts a {direction} win probability mainly influenced by {top_feats[0]} and {top_feats[1]}."

            # --- Store results ---
            explanation.update({
                "feature_importance": shap_imp,
                "steps": steps,
                "summary": summary,
                "plot_path": str(waterfall_path),
                "plot_base64": img_b64
            })

        except Exception as e:
            explanation["error"] = f"SHAP explainability failed: {str(e)}"

    else:
        explanation["note"] = "SHAP not available; baseline prediction used."

    return prob_p_win, explanation
