from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    session_id: str = Field(..., min_length=1, max_length=128)


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class HealthResponse(BaseModel):
    status: str
    service: str
    rag_loaded: bool