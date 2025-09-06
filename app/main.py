from fastapi import FastAPI
from app.schemas import TicketRequest, TicketResponse
from app.crypto import generate_ticket, verify_ticket
import hashlib

app = FastAPI(title="CTIA V1", description="Cryptographic Ticket Issuer Agent")

# Temporary in-memory store for tickets (will move to DB later)
tickets_db = {}
secrets_db = {}  # stores secrets for redemption (not shared with merchant)

@app.get("/")
def root():
    return {"message": "CTIA is running 🚀"}

@app.post("/issue_ticket", response_model=TicketResponse)
def issue_ticket(request: TicketRequest):
    # Generate a real signed ticket
    ticket, secret = generate_ticket(request.value, "recipient@upi")

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

    ticket["status"] = "redeemed"
    return {"message": f"Ticket {ticket_id} redeemed successfully"}

# ⚠️ Debug-only: get the secret (preimage) for a ticket
@app.get("/get_secret/{ticket_id}")
def get_secret(ticket_id: str):
    if ticket_id not in secrets_db:
        return {"error": "Secret not found"}
    return {"ticket_id": ticket_id, "secret": secrets_db[ticket_id]}
