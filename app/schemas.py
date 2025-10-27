# app/schemas.py
from pydantic import BaseModel
from typing import Optional, Dict

class TicketRequest(BaseModel):
    value: int

class TicketResponse(BaseModel):
    ticket_id: str
    value: int
    status: str
    prob_p_win: float
    lock_hash: str
    vpi_enc: str
    signature: str
    explanation: Optional[dict] = None

    class Config:
        from_attributes = True