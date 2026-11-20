import hmac
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage

from src.api_schemas import ChatRequest, ChatResponse, HealthResponse
from src.rag_pipeline import rag_chain
from src.session_manager import session_manager

load_dotenv()

app = FastAPI(
    title="Advanced RAG API",
    description="REST API for the existing LangChain Advanced RAG pipeline.",
    version="1.0.0",
)

# Set CORS_ORIGINS to comma-separated trusted frontend origins in production.
origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


def verify_api_key(authorization: Optional[str] = Header(default=None)) -> None:
    expected_key = os.getenv("RAG_API_KEY")
    if not expected_key:
        return  # Convenient for local development; configure in Railway.
    expected_header = f"Bearer {expected_key}"
    if not authorization or not hmac.compare_digest(authorization, expected_header):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/")
def root():
    return {
        "service": "Advanced RAG API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        service="advanced-rag-api",
        rag_loaded=True,
    )


@app.post(
    "/api/rag/chat",
    response_model=ChatResponse,
    dependencies=[Depends(verify_api_key)],
)
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    history = session_manager.get_history(request.session_id)

    try:
        answer = rag_chain.invoke({
            "input": question,
            "chat_history": history[-10:],
        })
        answer = answer if isinstance(answer, str) else str(answer)
    except Exception:
        # Avoid exposing provider credentials or internal stack traces.
        raise HTTPException(
            status_code=500,
            detail="RAG processing failed. Check server logs and configuration.",
        )

    session_manager.append_turn(
        request.session_id,
        HumanMessage(content=question),
        AIMessage(content=answer),
    )

    return ChatResponse(answer=answer, session_id=request.session_id)


@app.delete(
    "/api/rag/session/{session_id}",
    dependencies=[Depends(verify_api_key)],
)
def clear_session(session_id: str):
    cleared = session_manager.clear(session_id)
    return {"session_id": session_id, "cleared": cleared}
