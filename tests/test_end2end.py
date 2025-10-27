import pytest
import json
import os
import subprocess
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.main import app, get_db
from app.db import SessionLocal
from app import models

client = TestClient(app)

def override_get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

EDA_SCRIPT = "notebooks/eda_and_train.py"
MODELS_DIR = "models/"

def test_issue_redeem_and_explain_ticket():
    """End-to-end API test: issue → redeem → explain"""
    response = client.post("/issue_ticket", json={"value": 10})
    assert response.status_code == 200
    ticket = response.json()
    ticket_id = ticket["ticket_id"]

    db: Session = next(override_get_db())
    secret = db.query(models.Secret).filter_by(ticket_id=ticket_id).first()
    assert secret is not None
    preimage = secret.secret

    redeem_response = client.post(f"/redeem_ticket?ticket_id={ticket_id}&preimage={preimage}")
    assert redeem_response.status_code == 200
    redeem_data = redeem_response.json()
    assert "winner" in redeem_data
    assert "message" in redeem_data

    explain_resp = client.get(f"/explain_ticket/{ticket_id}")
    assert explain_resp.status_code == 200
    ex = explain_resp.json()["explanation"]
    assert "final_prob" in ex
    assert isinstance(ex.get("steps", []), list)


def test_training_pipeline_end_to_end():
    """Ensure EDA + training pipeline runs and saves valid artifacts."""
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
    assert rmse is not None, f"RMSE missing in metadata → found keys: {list(metrics.keys())}"
    assert rmse < 0.1, f"Model RMSE too high — got {rmse:.4f}"

    print(f"✅ Model validated successfully: {meta['best_model']} (RMSE={rmse:.4f})")
