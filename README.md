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

ctia/
│── app/ # Core FastAPI app (main.py, schema.py, crypto.py)
│── tests/ # Unit & end-to-end tests
│── secrets/ # Ed25519 keys (auto-generated on first run)
│── ctia_fsm.dot # FSM diagram (Graphviz source)
│── fsm.png # FSM diagram (rendered)
│── requirements.txt # Python dependencies
│── Dockerfile # Container definition
│── docker-compose.yml # Compose setup for local dev
│── setup.sh # Quick setup script
│── README.md # This file
│── LICENSE # MIT License


---

## ⚡ Quickstart

Clone the repository:

```bash
git clone https://github.com/ctia-project/ctia.git
cd ctia

./setup.sh

uvicorn app.main:app --reload --port 8000

docker-compose up --build

