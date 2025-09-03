from fastapi.testclient import TestClient
from app.main import app, TICKETS

client = TestClient(app)

def test_issue_and_redeem():
    resp = client.post("/issue_batch?issuer=agent://issuer-1&n=1&payout=1.0")
    assert resp.status_code == 200
    t = resp.json()["tickets"][0]
    tid = t["ticket_id"]

    preimage = TICKETS[tid]["preimage"]
    r = client.post("/redeem", params={"ticket_id": tid, "preimage": preimage})
    assert r.status_code == 200
    assert r.json()["status"] == "redeemed"
