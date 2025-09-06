# 🔐 CTIA — Cryptographic Ticket Issuer Agent (Prototype)

This project implements a **Cryptographic Ticket Issuer Agent (CTIA)** that issues and redeems secure, verifiable micropayment tickets.  
It is a **research prototype** designed for evaluating cryptographic ticketing mechanisms in micropayment systems.  

Each ticket contains:
- 🆔 Unique ticket identifier  
- 🔒 Lock condition (hash-preimage)  
- 💰 INR payout value  
- 📩 Encrypted Virtual Payment Interface (VPI) with recipient’s UPI address (placeholder)  
- ✍️ Cryptographic signature (Ed25519)  

The system supports:
- ✅ Probabilistic validation of tickets  
- ✅ Issuance and redemption flows  
- ✅ JSON-based canonical signing  
- ✅ Forward-compatibility with **X402-compliant envelopes** (future work)  

---

## 📂 Project Structure

```text
ctia/
│── app/                # Core FastAPI app (main.py, schema.py, crypto.py)
│   │── __init__.py
│   │── main.py         # FastAPI entrypoint (issue/redeem endpoints)
│   │── schema.py       # Pydantic models for ticket structure
│   │── crypto.py       # Ed25519 signing & verification helpers
│
│── tests/              # Unit & end-to-end tests
│   │── test_end2end.py
│
│── secrets/            # Auto-generated keys (ed25519_sk.hex, ed25519_pk.hex)
│
│── ctia_fsm.dot        # FSM definition (Graphviz source)
│── fsm.png             # Rendered FSM diagram
│── requirements.txt    # Python dependencies (frozen)
│── Dockerfile          # Container definition
│── docker-compose.yml  # Compose setup for local dev
│── setup.sh            # Quick setup script
│── README.md           # Documentation
│── LICENSE             # MIT License

```
---

## ⚡ Quickstart

Clone the repository:

```bash
git clone https://github.com/ctia-project/ctia.git
cd ctia

./setup.sh

uvicorn app.main:app --reload --port 8000

docker-compose up --build

