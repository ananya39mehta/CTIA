# notebooks/eda_and_train.py
"""
EDA + Multi-Model Training + Comparison Visualization (CTIA v2.2)
-----------------------------------------------------------------
- Loads live telemetry dataset (or simulated if missing)
- Trains multiple ML models and compares performance
- Computes SHAP/permutation importance
- Visualizes model performance and predictions
- Saves best model for CTIA Explainable AI pipeline
"""

import os
import json
import time
import math
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# Optional imports
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_PATH = os.path.join(ROOT, "docs")
MODELS_DIR = os.path.join(ROOT, "models")
NOTEBOOKS_OUT = os.path.join(ROOT, "notebooks")
os.makedirs(DOCS_PATH, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(NOTEBOOKS_OUT, exist_ok=True)

DATA_PATH = os.path.join(DOCS_PATH, "live_telemetry_dataset.json")


# ----------------------------
# Load telemetry dataset
# ----------------------------
def load_telemetry_dataset(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No dataset found at {path}. Run generate_live_telemetry_dataset.py first.")
    with open(path, "r") as f:
        data = json.load(f)
    rows = []
    for r in data:
        inp = r.get("inputs", r)
        rows.append({
            "network_load": inp.get("network_load", np.nan),
            "relay_reputation": inp.get("relay_reputation", np.nan),
            "budget_utilization": inp.get("budget_utilization", np.nan),
            "recent_win_rate": inp.get("recent_win_rate", np.nan),
            "final_prob": r.get("final_prob", np.nan),
        })
    df = pd.DataFrame(rows).dropna()
    print(f"📡 Loaded dataset with {len(df)} samples from {path}")
    return df


# ----------------------------
# Train + Compare Models
# ----------------------------
def train_and_compare(df: pd.DataFrame):
    X = df[["network_load", "relay_reputation", "budget_utilization", "recent_win_rate"]]
    y = df["final_prob"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    candidates = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "svr": Pipeline([("scaler", StandardScaler()), ("svr", SVR(C=10, epsilon=0.01))]),
        "mlp_regressor": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    }
    if XGB_AVAILABLE:
        candidates["xgb_regressor"] = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)

    results = {}
    for name, model in candidates.items():
        print(f"🧠 Training {name} ...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
        results[name] = {"model": model, "rmse": rmse, "mae": mae, "r2": r2, "cv_r2": cv}
        print(f"  ✅ RMSE={rmse:.6f}, MAE={mae:.6f}, R2={r2:.4f}, CV_R2={cv:.4f}")

    comp = pd.DataFrame([{ "name": k, **{kk: vv for kk, vv in v.items() if kk != "model"} } for k, v in results.items()])
    comp.to_csv(os.path.join(NOTEBOOKS_OUT, "model_comparison.csv"), index=False)
    print("📊 Model comparison saved to notebooks/model_comparison.csv")

    # Plot model comparison
    plt.figure(figsize=(8,4))
    sns.barplot(x="rmse", y="name", data=comp.sort_values("rmse"))
    plt.title("Model RMSE Comparison (Lower is Better)")
    plt.tight_layout()
    plt.savefig(os.path.join(NOTEBOOKS_OUT, "model_comparison_rmse.png"))
    plt.close()

    plt.figure(figsize=(8,4))
    sns.barplot(x="r2", y="name", data=comp.sort_values("r2", ascending=False))
    plt.title("Model R² Comparison (Higher is Better)")
    plt.tight_layout()
    plt.savefig(os.path.join(NOTEBOOKS_OUT, "model_comparison_r2.png"))
    plt.close()

    best_name = min(results.keys(), key=lambda n: results[n]["rmse"])
    best_model = results[best_name]["model"]
    print(f"🏆 Best model: {best_name}")

    # Save best model
    model_path = os.path.join(MODELS_DIR, "probability_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"💾 Saved best model to {model_path}")

    # SHAP or permutation importance
    feature_importance = {}
    try:
        if SHAP_AVAILABLE:
            print("🔍 Computing SHAP feature importance...")
            expl = shap.Explainer(best_model.predict, X_train)
            shap_vals = expl(X_test)
            mean_abs_shap = np.mean(np.abs(shap_vals.values), axis=0)
            feature_importance = dict(zip(X.columns, mean_abs_shap.tolist()))
            shap.summary_plot(shap_vals, X_test, show=False)
            plt.savefig(os.path.join(NOTEBOOKS_OUT, "shap_summary.png"))
            plt.close()
        else:
            print("⚙️ Using permutation importance...")
            perm = permutation_importance(best_model, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1)
            feature_importance = dict(zip(X.columns, perm.importances_mean.tolist()))
    except Exception as e:
        print("⚠️ Explainability computation failed:", e)

    # Feature importance plot
    if feature_importance:
        fi_sorted = sorted(feature_importance.items(), key=lambda x: -abs(x[1]))
        names, vals = zip(*fi_sorted)
        plt.figure(figsize=(6,3))
        sns.barplot(x=list(vals), y=list(names))
        plt.title("Feature Importance (Best Model)")
        plt.tight_layout()
        plt.savefig(os.path.join(NOTEBOOKS_OUT, "feature_importance_best.png"))
        plt.close()

    # Save meta info (remove actual model objects for JSON)
    safe_results = {
        name: {k: v for k, v in res.items() if k != "model"}
        for name, res in results.items()
    }

    meta = {
        "best_model": best_name,
        "metrics": safe_results[best_name],
        "all_results": safe_results,
        "feature_importance": feature_importance,
        "timestamp": datetime.utcnow().isoformat(),
        "n_samples": len(df)
    }

    with open(os.path.join(MODELS_DIR, "model_info.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("🧾 Saved model_info.json")

    # ----------------------------
    # Visualization: Predictions of all models
    # ----------------------------
    sample_df = X_test.sample(min(50, len(X_test)), random_state=42)
    preds_compare = pd.DataFrame(index=sample_df.index)
    for name, model in results.items():
        preds_compare[name] = model["model"].predict(sample_df)

    preds_compare["actual"] = y_test.loc[sample_df.index].values

    plt.figure(figsize=(10,6))
    for name in candidates.keys():
        plt.plot(preds_compare["actual"], preds_compare[name], 'o', alpha=0.5, label=name)
    plt.xlabel("Actual Final Probability")
    plt.ylabel("Predicted Final Probability")
    plt.title("Predicted vs Actual (All Models)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(NOTEBOOKS_OUT, "prediction_comparison_all_models.png"))
    plt.close()

    print("📈 Generated all comparison visualizations.")
    return best_name, meta


# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    df = load_telemetry_dataset()
    best_model, meta = train_and_compare(df)
    print("\n✅ Training complete.")
    print(f"🏆 Best Model: {best_model}")
    print(f"📂 Artifacts in: notebooks/ and models/")
