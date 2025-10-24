"""
notebooks/eda_and_train.py
- Loads docs/simulated_decisions.json and docs/decisions_log.json (if available)
- Performs basic EDA and saves summary + plots to notebooks/
- Trains several models (LinearRegression, DecisionTree, RandomForest)
- Evaluates models and selects the best by RMSE (or R2)
- Saves the best model to models/probability_model.pkl
- Computes SHAP explanations if shap is installed; otherwise uses permutation importance
- Saves model metadata to models/model_info.json
"""

import os
import json
import joblib
import math
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_PATH = os.path.join(ROOT, "docs")
NOTEBOOKS_OUT = os.path.join(ROOT, "notebooks")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(NOTEBOOKS_OUT, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------- 1. Load dataset ----------
def load_simulated_and_logs(sim_path="docs/simulated_decisions.json", logs_path="docs/decisions_log.json"):
    rows = []
    # load simulated dataset if exists
    if os.path.exists(sim_path):
        with open(sim_path) as f:
            sim = json.load(f)
        for r in sim:
            inp = r.get("inputs") or {}
            rows.append({
                "network_load": float(inp.get("network_load", np.nan)),
                "relay_reputation": float(inp.get("relay_reputation", np.nan)),
                "budget_utilization": float(inp.get("budget_utilization", np.nan)),
                "recent_win_rate": float(inp.get("recent_win_rate", np.nan)),
                "final_prob": float(r.get("final_prob", r.get("prob_p_win", np.nan)))
            })
    # load decision logs if exists (real runs)
    if os.path.exists(logs_path):
        with open(logs_path) as f:
            logs = json.load(f)
        for r in logs:
            inp = r.get("inputs") or r.get("explanation", {}).get("inputs", {})
            final = r.get("final_prob") or r.get("prob_p_win") or r.get("explanation", {}).get("final_prob")
            rows.append({
                "network_load": float(inp.get("network_load", np.nan)),
                "relay_reputation": float(inp.get("relay_reputation", np.nan)),
                "budget_utilization": float(inp.get("budget_utilization", np.nan)),
                "recent_win_rate": float(inp.get("recent_win_rate", np.nan)),
                "final_prob": float(final)
            })
    df = pd.DataFrame(rows)
    return df

df = load_simulated_and_logs()
print(f"Loaded dataset: {len(df)} rows")
print(df.head())

# ---------- 2. Basic EDA ----------
desc = df.describe().transpose()
desc.to_csv(os.path.join(NOTEBOOKS_OUT, "eda_summary.csv"))
print("Saved eda_summary.csv")

# Correlation matrix
corr = df.corr()
corr.to_csv(os.path.join(NOTEBOOKS_OUT, "correlation_matrix.csv"))

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation")
plt.tight_layout()
plt.savefig(os.path.join(NOTEBOOKS_OUT, "correlation_matrix.png"))
plt.close()

# Histograms
df.hist(bins=30, figsize=(10,8))
plt.tight_layout()
plt.savefig(os.path.join(NOTEBOOKS_OUT, "histograms.png"))
plt.close()

# Scatter: final_prob vs features
for col in ["network_load", "relay_reputation", "budget_utilization", "recent_win_rate"]:
    plt.figure(figsize=(6,3))
    plt.scatter(df[col], df["final_prob"], alpha=0.5)
    plt.xlabel(col)
    plt.ylabel("final_prob")
    plt.title(f"final_prob vs {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(NOTEBOOKS_OUT, f"scatter_{col}.png"))
    plt.close()

# ---------- 3. Prepare training data ----------
df = df.dropna()
X = df[["network_load", "relay_reputation", "budget_utilization", "recent_win_rate"]].astype(float)
y = df["final_prob"].astype(float)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ---------- 4. Train candidate models ----------
models = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
    results[name] = {"model": model, "rmse": rmse, "mae": mae, "r2": r2, "cv_r2": cv}
    print(f"  RMSE={rmse:.6f}, MAE={mae:.6f}, R2={r2:.4f}, CV_R2={cv:.4f}")

# Save model comparison
comp = pd.DataFrame([{ "name": k, **{kk: vv for kk, vv in v.items() if kk != "model"} } for k, v in results.items()])
comp.to_csv(os.path.join(NOTEBOOKS_OUT, "model_comparison.csv"), index=False)

# Choose best model by RMSE (lower better) or R2 (higher better). Here we use RMSE.
best_name = min(results.keys(), key=lambda n: results[n]["rmse"])
best_model = results[best_name]["model"]
print(f"Best model by RMSE: {best_name}")

# Save best model
model_path = os.path.join(MODELS_DIR, "probability_model.pkl")
joblib.dump(best_model, model_path)
print(f"Saved best model to {model_path}")

# ---------- 5. Explain the model ----------
explain_out = {}
# If SHAP available and model supports shap (tree or linear), compute SHAP values
if SHAP_AVAILABLE:
    try:
        print("Computing SHAP explanations (may take a moment)...")
        # shap for tree-based models will be fast; for pipelines, pass through X_train
        if hasattr(best_model, "predict") and isinstance(best_model, Pipeline):
            # pipeline: extract last estimator and use original X_train scaled
            expl = shap.Explainer(best_model.predict, X_train)
            shap_vals = expl(X_test)
        else:
            expl = shap.Explainer(best_model, X_train)
            shap_vals = expl(X_test)
        # summarize
        mean_abs_shap = np.mean(np.abs(shap_vals.values), axis=0)
        feature_names = X_train.columns.tolist()
        feat_importance = dict(zip(feature_names, mean_abs_shap.tolist()))
        explain_out["method"] = "shap"
        explain_out["feature_importance"] = feat_importance
        # Save a SHAP summary plot
        try:
            shap.summary_plot(shap_vals, X_test, show=False)
            plt.savefig(os.path.join(NOTEBOOKS_OUT, "shap_summary.png"))
            plt.close()
        except Exception:
            pass
    except Exception as e:
        print("SHAP failed:", e)
        SHAP_AVAILABLE = False

if not SHAP_AVAILABLE:
    print("SHAP not available — using permutation importance.")
    perm = permutation_importance(best_model, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)
    feature_names = X_train.columns.tolist()
    feat_importance = dict(zip(feature_names, perm.importances_mean.tolist()))
    explain_out["method"] = "permutation"
    explain_out["feature_importance"] = feat_importance

# Save feature importance
with open(os.path.join(MODELS_DIR, "model_info.json"), "w") as f:
    meta = {
        "best_model": best_name,
        "rmse": results[best_name]["rmse"],
        "mae": results[best_name]["mae"],
        "r2": results[best_name]["r2"],
        "cv_r2": results[best_name]["cv_r2"],
        "timestamp": datetime.utcnow().isoformat(),
        "explain": explain_out
    }
    json.dump(meta, f, indent=4)
print("Saved model_info.json")

# Save feature importance figure (simple bar)
fi_items = sorted(explain_out["feature_importance"].items(), key=lambda x: -abs(x[1]))
names = [x[0] for x in fi_items]
vals = [x[1] for x in fi_items]
plt.figure(figsize=(6,3))
sns.barplot(x=vals, y=names)
plt.title("Feature importance")
plt.tight_layout()
plt.savefig(os.path.join(NOTEBOOKS_OUT, "feature_importance.png"))
plt.close()

print("All done. Outputs in notebooks/ and models/")
