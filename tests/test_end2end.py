# tests/test_end2end.py
import pytest
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

def test_issue_and_redeem_ticket():
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
