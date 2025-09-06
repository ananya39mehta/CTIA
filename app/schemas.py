from pydantic import BaseModel

class TicketRequest(BaseModel):
    value: int

class TicketResponse(BaseModel):
    ticket_id: str
    value: int
    status: str
