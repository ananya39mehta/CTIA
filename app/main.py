from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app import models
from app.db import SessionLocal, init_db
from app.schemas import TicketRequest, TicketResponse
from app.crypto import generate_ticket, verify_ticket, is_winner
from app.agent_ml import decide_probability

import hashlib, json, os

# Initialize DB
init_db()

app = FastAPI(title="CTIA v1.1", description="Autonomous & Explainable Ticket Issuer")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def root():
    return {"message": "CTIA v1.1 is running 🚀"}


@app.post("/issue_ticket", response_model=TicketResponse)
def issue_ticket(request: TicketRequest, db: Session = Depends(get_db)):
    # --- Simulated runtime inputs (replace with live metrics later)
    inputs = {
        "network_load": 0.3,
        "relay_reputation": 0.8,
        "budget_utilization": 0.2,
        "recent_win_rate": 0.05
    }

    # --- Decide probability autonomously
    prob_p_win, explanation = decide_probability(inputs)

    # --- Generate ticket
    ticket_dict, secret = generate_ticket(request.value, "recipient@upi", prob_p_win)

    # --- Persist to DB
    db_ticket = models.Ticket(
        ticket_id=ticket_dict["ticket_id"],
        value=ticket_dict["value"],
        prob_p_win=prob_p_win,
        lock_hash=ticket_dict["lock_hash"],
        vpi_enc=ticket_dict["vpi_enc"],
        signature=ticket_dict["signature"],
        status="issued"
    )
    db.add(db_ticket)
    db.add(models.Secret(ticket_id=db_ticket.ticket_id, secret=secret))
    db.add(models.DecisionLog(ticket_id=db_ticket.ticket_id, explanation=json.dumps(explanation)))
    db.commit()

    # --- Save explanation to docs/decisions_log.json
    os.makedirs("docs", exist_ok=True)
    log_path = os.path.join("docs", "decisions_log.json")
    try:
        logs = json.load(open(log_path)) if os.path.exists(log_path) else []
    except json.JSONDecodeError:
        logs = []
    logs.append({
        "ticket_id": db_ticket.ticket_id,
        "inputs": inputs,
        "prob_p_win": prob_p_win,
        "explanation": explanation
    })
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=4)

    # --- Return clean response
    rationale = "; ".join([s["effect"] for s in explanation["steps"]])
    return {
        "ticket_id": db_ticket.ticket_id,
        "value": db_ticket.value,
        "status": db_ticket.status,
        "prob_p_win": prob_p_win,
        "lock_hash": db_ticket.lock_hash,
        "vpi_enc": db_ticket.vpi_enc,
        "signature": db_ticket.signature,
        "explanation": {
            "final_prob": explanation["final_prob"],
            "rationale": rationale,
            "steps": explanation["steps"]
        }
    }


@app.post("/redeem_ticket")
def redeem_ticket(ticket_id: str, preimage: str, db: Session = Depends(get_db)):
    db_ticket = db.query(models.Ticket).filter_by(ticket_id=ticket_id).first()
    if not db_ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    if db_ticket.status == "redeemed":
        raise HTTPException(status_code=400, detail="Already redeemed")

    # Verify signature and preimage
    ticket_for_verify = {
        "ticket_id": db_ticket.ticket_id,
        "value": db_ticket.value,
        "lock_hash": db_ticket.lock_hash,
        "vpi_enc": db_ticket.vpi_enc,
        "prob_p_win": db_ticket.prob_p_win,
        "signature": db_ticket.signature
    }
    if not verify_ticket(ticket_for_verify):
        raise HTTPException(status_code=400, detail="Invalid signature")
    if hashlib.sha256(preimage.encode()).hexdigest() != db_ticket.lock_hash:
        raise HTTPException(status_code=400, detail="Invalid preimage")

    # Check winner
    winner = is_winner(ticket_for_verify, preimage)
    db.add(models.Redemption(ticket_id=ticket_id, preimage=preimage, winner=winner))
    db_ticket.status = "redeemed"
    db.commit()

    return {
        "message": f"Ticket {ticket_id} {'WON' if winner else 'lost'}.",
        "winner": winner,
        "payout": db_ticket.value if winner else 0
    }


@app.get("/explain_ticket/{ticket_id}")
def explain_ticket(ticket_id: str, db: Session = Depends(get_db)):
    dec = db.query(models.DecisionLog).filter_by(ticket_id=ticket_id).first()
    if not dec:
        raise HTTPException(status_code=404, detail="Explanation not found")
    return {"ticket_id": ticket_id, "explanation": json.loads(dec.explanation)}
