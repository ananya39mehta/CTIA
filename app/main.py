from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app import models
from app.db import SessionLocal, init_db
from app.schemas import TicketRequest, TicketResponse
from app.crypto import generate_ticket, verify_ticket, is_winner
import hashlib

# Initialize DB (create tables)
init_db()

app = FastAPI(title="CTIA V1", description="Cryptographic Ticket Issuer Agent with SQLite persistence")

# Dependency for DB sessions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {"message": "CTIA is running 🚀"}

@app.post("/issue_ticket", response_model=TicketResponse)
def issue_ticket(request: TicketRequest, db: Session = Depends(get_db)):
    # Generate a real signed ticket and secret (preimage)
    ticket_dict, secret = generate_ticket(request.value, "recipient@upi", request.prob_p_win)

    # Persist the ticket into DB
    db_ticket = models.Ticket(
        ticket_id=ticket_dict["ticket_id"],
        value=ticket_dict["value"],
        prob_p_win=float(ticket_dict.get("prob_p_win", 0.001)),
        lock_hash=ticket_dict["lock_hash"],
        vpi_enc=ticket_dict["vpi_enc"],
        signature=ticket_dict["signature"],
        status="issued"
    )
    db.add(db_ticket)
    db.commit()
    db.refresh(db_ticket)

    # Persist the secret for testing/debug (remove in production)
    db_secret = models.Secret(ticket_id=db_ticket.ticket_id, secret=secret)
    db.add(db_secret)
    db.commit()

    # Return a response matching TicketResponse schema
    return {
        "ticket_id": db_ticket.ticket_id,
        "value": db_ticket.value,
        "status": db_ticket.status,
        "prob_p_win": db_ticket.prob_p_win,
        "lock_hash": db_ticket.lock_hash,
        "vpi_enc": db_ticket.vpi_enc,
        "signature": db_ticket.signature,
    }

@app.post("/redeem_ticket")
def redeem_ticket(ticket_id: str, preimage: str, db: Session = Depends(get_db)):
    # Fetch ticket
    db_ticket = db.query(models.Ticket).filter_by(ticket_id=ticket_id).first()
    if not db_ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    if db_ticket.status == "redeemed":
        raise HTTPException(status_code=400, detail="Ticket already redeemed")

    # Reconstruct ticket dict for verification
    ticket_for_verify = {
        "ticket_id": db_ticket.ticket_id,
        "value": db_ticket.value,
        "lock_hash": db_ticket.lock_hash,
        "vpi_enc": db_ticket.vpi_enc,
        "prob_p_win": db_ticket.prob_p_win,
        "signature": db_ticket.signature
    }

    # Verify signature
    if not verify_ticket(ticket_for_verify):
        raise HTTPException(status_code=400, detail="Invalid ticket signature")

    # Verify preimage
    if hashlib.sha256(preimage.encode()).hexdigest() != db_ticket.lock_hash:
        raise HTTPException(status_code=400, detail="Invalid preimage (cannot unlock ticket)")

    # Determine winner deterministically
    winner = is_winner(ticket_for_verify, preimage)

    # Record redemption
    redemption = models.Redemption(ticket_id=db_ticket.ticket_id, preimage=preimage, winner=winner)
    db.add(redemption)

    # Mark ticket redeemed (prevents replay)
    db_ticket.status = "redeemed"
    db.commit()
    db.refresh(db_ticket)

    if winner:
        return {"message": f"Ticket {ticket_id} redeemed and is a WINNER. Initiating settlement.", "winner": True, "payout": db_ticket.value}
    else:
        return {"message": f"Ticket {ticket_id} redeemed but NOT a winner. No settlement required.", "winner": False}


