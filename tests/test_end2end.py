import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.main import app, get_db
from app.db import SessionLocal
from app import models

client = TestClient(app)

# Override get_db for tests to use same SessionLocal
def override_get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db


def test_issue_and_redeem_ticket():
    # Step 1: Issue a ticket
    response = client.post("/issue_ticket", json={"value": 10, "prob_p_win": 0.5})
    assert response.status_code == 200
    ticket = response.json()

    ticket_id = ticket["ticket_id"]

    # Step 2: Fetch the secret directly from DB
    db: Session = next(override_get_db())
    secret = db.query(models.Secret).filter_by(ticket_id=ticket_id).first()
    assert secret is not None
    preimage = secret.secret

    # Step 3: Redeem the ticket with the preimage
    redeem_response = client.post(
        f"/redeem_ticket?ticket_id={ticket_id}&preimage={preimage}"
    )
    assert redeem_response.status_code == 200
    redeem_data = redeem_response.json()

    # Step 4: Check response fields
    assert "winner" in redeem_data
    assert "message" in redeem_data

