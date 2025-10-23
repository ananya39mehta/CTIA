from pydantic import BaseModel
from typing import Optional

# Request schema
class TicketRequest(BaseModel):
    value: int

# Response schema
class TicketResponse(BaseModel):
    ticket_id: str
    value: int
    status: str
    prob_p_win: float
    lock_hash: str
    vpi_enc: str
    signature: str
    explanation: Optional[dict] = None  # Added for explainable AI responses

    class Config:
        from_attributes = True

