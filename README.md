# 🔐 CTIA — Cryptographic Ticket Issuer Agent (Prototype)

## 🌍 Overview

**CTIA v2** simulates a **trustless micropayment ecosystem** where each transaction (a “ticket”) is assigned a *dynamic winning probability* using **Explainable AI (XAI)** models trained on **system telemetry**.

This project demonstrates how **AI-driven probability models** can:
- Adjust in real time based on system conditions  
- Explain each decision transparently using SHAP and visual reasoning  
- Support ethical, auditable, and secure decision-making  

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

## 🎯 Objectives

| Goal | Description |
|------|--------------|
| **1. Explainable Probability Modeling** | Build an interpretable ML system for win/loss prediction using telemetry. |
| **2. Realistic System Simulation** | Generate telemetry data representing network, budget, and reputation. |
| **3. Trustless Micropayment Logic** | Use cryptographic hash locks for ticket issuance and redemption. |
| **4. Ethical AI Governance** | Maintain full auditability of every AI decision. |
| **5. Integration & Testing** | Provide reproducible code, datasets, and automated testing suite. |

---

## 🧩 System Architecture

```mermaid
flowchart TD

A["Telemetry Simulation"] -->|"Network Load, Budget, Reputation, Win Rate"| B["ML Model"]
B -->|"Predicts Win Probability"| C["Explainable AI (SHAP)"]
C -->|"Feature Importance + Visuals"| D["Decision Log"]
D -->|"Stores JSON Log + Plot"| E["FastAPI App"]
E -->|"/issue_ticket"| F["Ticket Issuance"]
E -->|"/redeem_ticket"| G["Redemption Validation"]
E -->|"/explain_ticket_visual"| H["Explainability Dashboard"]
H -->|"HTML/Plot Output"| I["User Interface"]

---

## 📂 Project Structure

```text
ctia/
│
├── app/
│   ├── main.py                  # FastAPI API endpoints
│   ├── models.py                # SQLAlchemy ORM models
│   ├── crypto/                  # Cryptographic ticket generation & verification
│   ├── agent_ml.py              # ML model loading, inference, SHAP explainability
│   ├── telemetry.py             # Telemetry simulation (network, reputation, budget)
│   └── db.py                    # Database initialization
│
├── data/
│   ├── raw/                     # Simulated telemetry data
│   └── processed/               # Cleaned datasets for model training
│
├── models/                      # Trained ML models (.pkl)
│
├── notebooks/
│   ├── eda_and_train.ipynb      # EDA + model training notebook
│
├── tests/
│   ├── test_end2end.py          # Tests issue→redeem→explain flow
│   ├── test_training_pipeline.py
│
├── docs/
│   ├── shap_explanations/       # Stored SHAP plots for explainability
│   ├── decisions_log.json       # Decision audit trail
│
├── requirements.txt
├── LICENSE
└── README.md


```
---

## ⚡ Quickstart

Clone the repository:

```bash
git clone https://github.com/ananya39mehta/CTIA.git
cd ctia
./setup.sh
uvicorn app.main:app --reload --port 8000

```
| Task                         | Command                                          |
| ---------------------------- | ------------------------------------------------ |
| Initialize database          | `PYTHONPATH=. python app/db_init.py`             |
| Run app                      | `uvicorn app.main:app --reload`                  |
| Run tests                    | `PYTHONPATH=. pytest -v tests/`                  |
| Train model                  | `PYTHONPATH=. python notebooks/eda_and_train.py` |
| Generate SHAP explainability | `python app/agent_ml.py`                         |


