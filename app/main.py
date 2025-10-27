# app/main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app import models
from app.db import SessionLocal, init_db
from app.schemas import TicketRequest, TicketResponse
from app.crypto import generate_ticket, verify_ticket, is_winner
from app.agent_ml import decide_probability
from app.telemetry import get_telemetry  # NEW IMPORT
import hashlib, json, os
from fastapi.responses import HTMLResponse
from datetime import datetime
init_db()
app = FastAPI(title="CTIA v2.0 ML", description="CTIA with ML-based explainable probability")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {"message": "CTIA v2.0 ML is running 🚀"}

@app.post("/issue_ticket", response_model=TicketResponse)
def issue_ticket(request: TicketRequest, db: Session = Depends(get_db)):
    """
    Issue a probabilistic ticket with ML-determined win probability.
    Uses real-time telemetry for dynamic decision-making.
    """
    # Get real telemetry
    telemetry_service = get_telemetry(db)
    inputs = telemetry_service.get_all_metrics()
    
    # Log telemetry for debugging (FIXED: show as decimals matching ML input)
    print(f"\n{'='*60}")
    print(f"📊 Issuing Ticket - Telemetry:")
    print(f"   Network Load: {inputs['network_load']:.2f} ({inputs['network_load']*100:.0f}%)")
    print(f"   Relay Reputation: {inputs['relay_reputation']:.2f} ({inputs['relay_reputation']*100:.0f}%)")
    print(f"   Budget Utilization: {inputs['budget_utilization']:.2f} ({inputs['budget_utilization']*100:.0f}%)")
    print(f"   Recent Win Rate: {inputs['recent_win_rate']:.3f} ({inputs['recent_win_rate']*100:.1f}%)")
    print(f"{'='*60}\n")
    
    # ML decision with real telemetry
    prob_p_win, explanation = decide_probability(inputs, telemetry=inputs)
    
    # Generate cryptographic ticket
    ticket_dict, secret = generate_ticket(request.value, "recipient@upi", prob_p_win)

    # Store in database
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
    
    # Store explanation with timestamp
    explanation['timestamp'] = datetime.utcnow().isoformat()
    explanation['telemetry'] = inputs
    db.add(models.DecisionLog(ticket_id=db_ticket.ticket_id, explanation=json.dumps(explanation)))
    db.commit()

    # Append to docs log
    os.makedirs("docs", exist_ok=True)
    log_path = os.path.join("docs", "decisions_log.json")
    try:
        existing = json.load(open(log_path)) if os.path.exists(log_path) else []
    except Exception:
        existing = []
    
    existing.append({
        "ticket_id": db_ticket.ticket_id,
        "value": db_ticket.value,
        "telemetry": inputs,
        "prob_p_win": prob_p_win,
        "explanation": explanation,
        "timestamp": explanation.get("timestamp")
    })
    
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=4)

    # Build rationale
    rationale = ""
    if isinstance(explanation, dict):
        steps = explanation.get("steps", [])
        if steps:
            try:
                rationale = "; ".join(str(s.get("effect", "")) for s in steps if s)
            except Exception:
                rationale = ""
        else:
            fi = explanation.get("feature_importance", {})
            if isinstance(fi, dict) and fi:
                rationale = "; ".join(f"{k}: importance={round(float(v), 6)}" for k, v in fi.items())
            else:
                rationale = "No detailed steps available."
    else:
        rationale = str(explanation)

    return {
        "ticket_id": db_ticket.ticket_id,
        "value": db_ticket.value,
        "status": db_ticket.status,
        "prob_p_win": prob_p_win,
        "lock_hash": db_ticket.lock_hash,
        "vpi_enc": db_ticket.vpi_enc,
        "signature": db_ticket.signature,
        "explanation": {
            "final_prob": explanation.get("final_prob"),
            "rationale": rationale,
            "steps": explanation.get("steps", []),
            "feature_importance": explanation.get("feature_importance", {}),
            "telemetry": inputs
        }
    }
# ... rest of your endpoints remain the same ...

@app.post("/redeem_ticket")
def redeem_ticket(ticket_id: str, preimage: str, db: Session = Depends(get_db)):
    db_ticket = db.query(models.Ticket).filter_by(ticket_id=ticket_id).first()
    if not db_ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    if db_ticket.status == "redeemed":
        raise HTTPException(status_code=400, detail="Ticket already redeemed")

    ticket_for_verify = {
        "ticket_id": db_ticket.ticket_id,
        "value": db_ticket.value,
        "lock_hash": db_ticket.lock_hash,
        "vpi_enc": db_ticket.vpi_enc,
        "prob_p_win": db_ticket.prob_p_win,
        "signature": db_ticket.signature
    }

    if not verify_ticket(ticket_for_verify):
        raise HTTPException(status_code=400, detail="Invalid ticket signature")
    if hashlib.sha256(preimage.encode()).hexdigest() != db_ticket.lock_hash:
        raise HTTPException(status_code=400, detail="Invalid preimage (cannot unlock ticket)")

    winner = is_winner(ticket_for_verify, preimage)
    db.add(models.Redemption(ticket_id=db_ticket.ticket_id, preimage=preimage, winner=winner))
    db_ticket.status = "redeemed"
    db.commit()

    if winner:
        return {"message": f"Ticket {ticket_id} redeemed and is a WINNER. Initiating settlement.", "winner": True, "payout": db_ticket.value}
    else:
        return {"message": f"Ticket {ticket_id} redeemed but NOT a winner. No settlement required.", "winner": False}


@app.get("/explain_ticket/{ticket_id}")
def explain_ticket(ticket_id: str, db: Session = Depends(get_db)):
    dec = (
        db.query(models.DecisionLog)
        .filter_by(ticket_id=ticket_id)
        .order_by(models.DecisionLog.decision_time.desc())
        .first()
    )
    if not dec:
        raise HTTPException(status_code=404, detail="Explanation not found")

    explanation = json.loads(dec.explanation)
    return {
        "ticket_id": ticket_id,
        "explanation": explanation,
        "plot_image_base64": explanation.get("plot_base64", None),
    }


@app.get("/explain_ticket_visual/{ticket_id}", response_class=HTMLResponse)
def explain_ticket_visual(ticket_id: str, db: Session = Depends(get_db)):
    dec = db.query(models.DecisionLog).filter_by(ticket_id=ticket_id).first()
    if not dec:
        return HTMLResponse("<h3>Explanation not found</h3>", status_code=404)

    exp = json.loads(dec.explanation)
    img_b64 = exp.get("plot_base64", "")
    telemetry = exp.get("telemetry", {})
    feature_importance = exp.get("feature_importance", {})

    # ---------- Telemetry color coding ----------
    def colorize(value: float, low: float, high: float):
        """Return emoji and color span based on thresholds"""
        if value < low:
            return f"<span style='color:green;'>🟢 {value:.2%}</span>"
        elif value > high:
            return f"<span style='color:red;'>🔴 {value:.2%}</span>"
        else:
            return f"<span style='color:orange;'>🟡 {value:.2%}</span>"

    telemetry_rows = []
    if telemetry:
        telemetry_rows.append("<tr><th>Metric</th><th>Value</th></tr>")
        for k, v in telemetry.items():
            if "network_load" in k:
                col = colorize(v, 0.3, 0.7)
            elif "relay_reputation" in k:
                col = colorize(1 - v, 0.3, 0.7)  # higher is better
            elif "budget" in k:
                col = colorize(v, 0.5, 0.8)
            elif "win_rate" in k:
                col = colorize(v, 0.01, 0.03)
            else:
                col = f"{v:.2%}"
            telemetry_rows.append(f"<tr><td>{k.replace('_', ' ').title()}</td><td>{col}</td></tr>")

    telemetry_table = "<table border='1' cellspacing='0' cellpadding='6'>" + "".join(telemetry_rows) + "</table>"

    # ---------- Feature importance summary ----------
    if feature_importance:
        top_features = sorted(feature_importance.items(), key=lambda x: -abs(x[1]))[:3]
        top_html = "<ul>" + "".join([
            f"<li><b>{k}</b>: influence <span style='color:#0077cc;'>{abs(v):.4f}</span></li>"
            for k, v in top_features
        ]) + "</ul>"
    else:
        top_html = "<p>No feature importance data available.</p>"

    # ---------- Build final HTML ----------
    html = f"""
    <html>
    <head>
        <title>Ticket {ticket_id} Explanation</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }}
            h2, h3 {{
                color: #333;
            }}
            table {{
                border-collapse: collapse;
                margin-bottom: 15px;
            }}
            th {{
                background-color: #f4f4f4;
            }}
        </style>
    </head>
    <body>
        <h2>🎟️ Ticket {ticket_id}</h2>

        <h3>📊 Telemetry at Decision Time</h3>
        {telemetry_table}

        <h3>🤖 ML Decision</h3>
        <p><b>Final Probability:</b> {exp['final_prob']:.4f} ({exp['final_prob']*100:.2f}%)</p>

        <h3>📈 Top Influencing Features</h3>
        {top_html}

        <h3>🧠 SHAP Explainability Plot</h3>
        <img src="data:image/png;base64,{img_b64}" width="700" style="border:1px solid #ddd; border-radius:10px;">
    </body>
    </html>
    """
    return HTMLResponse(html)


# NEW: Telemetry endpoint for monitoring
@app.get("/telemetry")
def get_current_telemetry(db: Session = Depends(get_db)):
    """
    Get current system telemetry snapshot.
    Useful for monitoring and debugging.
    """
    telemetry_service = get_telemetry(db)
    metrics = telemetry_service.get_all_metrics()
    summary = telemetry_service.get_telemetry_summary()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "summary": summary
    }