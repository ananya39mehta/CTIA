"""
app/agent_ml.py
- Loads models/probability_model.pkl (joblib)
- Exposes decide_probability(inputs: dict) -> (prob_p_win, explanation_dict)
- explanation_dict: { method, final_prob, inputs, feature_importance }
"""

import os, json
import numpy as np
import pandas as pd
import joblib

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "probability_model.pkl")
INFO_PATH = os.path.join(MODELS_DIR, "model_info.json")

# Try to load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run notebooks/eda_and_train.py first.")

model = joblib.load(MODEL_PATH)

# Try to load model_info for feature importances, fallback to empty
if os.path.exists(INFO_PATH):
    with open(INFO_PATH) as f:
        model_info = json.load(f)
else:
    model_info = {}

# Optional SHAP lazy importer (we compute shap explanations per-sample if shap installed)
try:
    import shap
    SHAP_AVAILABLE = True
    # create explainer if model is tree-based or pipeline: shap.Explainer can wrap the model
    try:
        explainer = shap.Explainer(model)
    except Exception:
        explainer = None
except Exception:
    SHAP_AVAILABLE = False
    explainer = None


def _to_dataframe(inputs: dict):
    # ensure columns order matches training columns used in eda_and_train.py
    cols = ["network_load", "relay_reputation", "budget_utilization", "recent_win_rate"]
    return pd.DataFrame([{c: float(inputs.get(c, 0.0)) for c in cols}])


def decide_probability(inputs: dict):
    """
    Return: (prob_p_win: float, explanation: dict)
    Normalized explanation contains:
      - method: "ml_model"
      - final_prob: float
      - inputs: dict
      - feature_importance: {feature: importance}
      - steps: [{feature, value, effect}]
    """
    X = _to_dataframe(inputs)

    # predict
    try:
        prob = float(model.predict(X)[0])
    except Exception:
        prob = float(model.predict(X.values)[0])

    explanation = {
        "method": "ml_model",
        "final_prob": prob,
        "inputs": inputs,
        "feature_importance": {}
    }

    # SHAP path: per-sample SHAP values -> abs contributions
    if SHAP_AVAILABLE and explainer is not None:
        try:
            sv = explainer(X)  # shap output
            # shap values may be structured; get per-feature absolute contributions
            shap_vals = sv.values[0] if hasattr(sv, "values") else np.array(sv)[0]
            feature_names = X.columns.tolist()
            feat_imp = {f: float(abs(v)) for f, v in zip(feature_names, shap_vals)}
            explanation["feature_importance"] = feat_imp
        except Exception:
            explanation["feature_importance"] = model_info.get("explain", {}).get("feature_importance", {})
    else:
        # fallback to saved permutation importances / metadata
        explanation["feature_importance"] = model_info.get("explain", {}).get("feature_importance", {})

    # Build human-readable 'steps' from feature_importance and inputs
    steps = []
    for f, imp in explanation["feature_importance"].items():
        val = float(inputs.get(f, 0.0))
        # craft a simple "effect" sentence using sign of importance and value
        # note: shap/permutation importance are non-negative; use value -> direction heuristics
        if f == "network_load":
            effect = f"higher {f} tends to {'decrease' if val>0.2 else 'slightly decrease' if val>0.1 else 'have minor effect'} prob"
        elif f == "relay_reputation":
            effect = f"higher {f} tends to increase prob" if val > 0.5 else f"{f} low -> lower prob"
        elif f == "budget_utilization":
            effect = f"higher {f} tends to decrease prob"
        elif f == "recent_win_rate":
            effect = f"higher {f} tends to decrease prob"
        else:
            effect = f"{f} effect (value={val})"
        steps.append({
            "feature": f,
            "value": val,
            "effect": effect,
            "importance": float(imp)
        })

    # Sort steps by importance desc (optional)
    steps = sorted(steps, key=lambda s: -s.get("importance", 0.0))

    # Attach steps and return
    explanation["steps"] = steps
    return prob, explanation
