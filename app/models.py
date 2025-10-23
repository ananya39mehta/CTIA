from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import relationship
from .db import Base
from datetime import datetime
from sqlalchemy import Text
import json
class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(String, unique=True, index=True, nullable=False)
    value = Column(Integer, nullable=False)
    prob_p_win = Column(Float, nullable=False, default=0.001)
    lock_hash = Column(String, nullable=False)
    vpi_enc = Column(Text, nullable=False)
    signature = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="issued")
    created_at = Column(DateTime, default=datetime.utcnow)
    expiry = Column(DateTime, nullable=True)

    redemptions = relationship("Redemption", back_populates="ticket", cascade="all, delete-orphan")


class Secret(Base):
    __tablename__ = "secrets"
    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(String, ForeignKey("tickets.ticket_id"), nullable=False, unique=True)
    secret = Column(String, nullable=False)


class Redemption(Base):
    __tablename__ = "redemptions"
    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(String, ForeignKey("tickets.ticket_id"), nullable=False)
    preimage = Column(String, nullable=False)
    winner = Column(Boolean, nullable=False)
    redeemed_at = Column(DateTime, default=datetime.utcnow)

    ticket = relationship("Ticket", back_populates="redemptions")



class DecisionLog(Base):
    __tablename__ = "decision_logs"
    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(String, ForeignKey("tickets.ticket_id"), nullable=True, index=True)
    decision_time = Column(DateTime, default=datetime.utcnow)
    explanation = Column(Text, nullable=False)  # store JSON string

    # optional relationship back to ticket
    ticket = relationship("Ticket", back_populates="decision_logs")

# add relationship on Ticket
if not hasattr(Ticket, "decision_logs"):
    Ticket.decision_logs = relationship("DecisionLog", back_populates="ticket", cascade="all, delete-orphan")
