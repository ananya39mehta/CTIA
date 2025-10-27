import os
import json
import subprocess

EDA_SCRIPT = "notebooks/eda_and_train.py"
MODELS_DIR = "models/"

def test_training_pipeline_end_to_end():
    """Verify EDA + training pipeline executes fully."""
    print("🧠 Running training pipeline...")
    result = subprocess.run(["python", EDA_SCRIPT], capture_output=True, text=True)
    print(result.stdout)
    assert result.returncode == 0, f"Training script failed: {result.stderr}"

    model_path = os.path.join(MODELS_DIR, "probability_model.pkl")
    info_path = os.path.join(MODELS_DIR, "model_info.json")

    assert os.path.exists(model_path), "❌ Model file not found"
    assert os.path.exists(info_path), "❌ Model info metadata missing"

    with open(info_path, "r") as f:
        meta = json.load(f)

    assert "best_model" in meta, "Metadata missing best_model key"

    # ✅ Handle lowercase nested metrics
    metrics = meta.get("metrics", {})
    rmse = (
        meta.get("best_rmse")
        or meta.get("rmse")
        or metrics.get("RMSE")
        or metrics.get("rmse")
    )
    assert rmse is not None, "RMSE missing in metadata"
    assert rmse < 0.1, f"Model RMSE too high — got {rmse:.4f}"

    print(f"✅ Model validated successfully: {meta['best_model']} (RMSE={rmse:.4f})")
