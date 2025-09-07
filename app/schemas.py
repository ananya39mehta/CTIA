from pydantic import BaseModel
from typing import Optional

class TicketRequest(BaseModel):
    value: int
    prob_p_win: Optional[float] = 0.001  # optional when client can set priority

class TicketResponse(BaseModel):
    ticket_id: str
    value: int
    status: str
    prob_p_win: float
    lock_hash: str
    vpi_enc: str
    signature: str
