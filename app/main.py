from fastapi import FastAPI
from app.schemas import TicketRequest, TicketResponse
from app.crypto import generate_ticket, verify_ticket, is_winner
import hashlib

app = FastAPI(title="CTIA V1", description="Cryptographic Ticket Issuer Agent")

# Temporary in-memory store for tickets and secrets (prototype)
tickets_db = {}
secrets_db = {}

@app.get("/")
def root():
    return {"message": "CTIA is running 🚀"}

@app.post("/issue_ticket", response_model=TicketResponse)
def issue_ticket(request: TicketRequest):
    # Generate a real signed ticket with optional probability
    ticket, secret = generate_ticket(request.value, "recipient@upi", request.prob_p_win)

    # Add status field required by schema
    ticket["status"] = "issued"

    tickets_db[ticket["ticket_id"]] = ticket
    secrets_db[ticket["ticket_id"]] = secret

    return ticket

@app.post("/redeem_ticket")
def redeem_ticket(ticket_id: str, preimage: str):
    if ticket_id not in tickets_db:
        return {"error": "Ticket not found"}

    ticket = tickets_db[ticket_id]

    # Verify Ed25519 signature
    if not verify_ticket(ticket):
        return {"error": "Invalid ticket signature"}

    # Verify preimage against lock_hash
    lock_hash = hashlib.sha256(preimage.encode()).hexdigest()
    if lock_hash != ticket["lock_hash"]:
        return {"error": "Invalid preimage (cannot unlock ticket)"}

    if ticket.get("status") == "redeemed":
        return {"error": "Ticket already redeemed"}

    # Deterministic probabilistic check
    winner = is_winner(ticket, preimage)

    # Mark redeemed (we keep status; real settlement only if winner)
    ticket["status"] = "redeemed"
    ticket["winner"] = winner

    if winner:
        # In prototype: simulate settlement
        return {"message": f"Ticket {ticket_id} redeemed and is a WINNER. Initiating settlement.", "winner": True, "payout": ticket["value"]}
    else:
        # Loser: no settlement on ledger; still mark redeemed to avoid replay
        return {"message": f"Ticket {ticket_id} redeemed but NOT a winner. No settlement required.", "winner": False}
    
# Debug-only: get the secret (preimage) for a ticket (testing only)
@app.get("/get_secret/{ticket_id}")
def get_secret(ticket_id: str):
    if ticket_id not in secrets_db:
        return {"error": "Secret not found"}
    return {"ticket_id": ticket_id, "secret": secrets_db[ticket_id]}
