
import os
import hmac
from typing import Dict, List

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from langchain_core.messages import HumanMessage, AIMessage

from src.rag_pipeline import rag_chain
from src.api_schemas import ChatRequest, ChatResponse, HealthResponse


# =========================================================
# 1. ENVIRONMENT CONFIGURATION
# =========================================================

load_dotenv(override=True)


# =========================================================
# 2. FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Advanced RAG API",
    description="Conversational Advanced RAG Chatbot using LangChain and FastAPI",
    version="1.0.0"
)


# =========================================================
# 3. CORS CONFIGURATION
# =========================================================

origins = os.getenv("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# 4. SWAGGER BEARER AUTHENTICATION
# =========================================================

security = HTTPBearer(auto_error=False)


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):

    expected_key = os.getenv("RAG_API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=500,
            detail="API key not configured"
        )

    if (
        credentials is None
        or not hmac.compare_digest(
            credentials.credentials,
            expected_key
        )
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )


# =========================================================
# 5. CONVERSATION HISTORY STORAGE
# =========================================================

store: Dict[str, List] = {}


# =========================================================
# 6. ROOT ROUTE
# =========================================================

@app.get("/")
def home():

    return {
        "service": "Advanced RAG API",
        "status": "running",
        "docs": "/docs"
    }


# =========================================================
# 7. HEALTH CHECK
# =========================================================

@app.get("/health", response_model=HealthResponse)
def health():

    return {
        "status": "healthy",
        "service": "advanced-rag-api",
        "rag_loaded": rag_chain is not None
    }


# =========================================================
# 8. CHAT API
# =========================================================

@app.post(
    "/api/rag/chat",
    response_model=ChatResponse,
    dependencies=[Depends(verify_api_key)]
)
def chat(request: ChatRequest):

    try:

        session_id = request.session_id
        question = request.question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )

        # Create session if it does not exist
        if session_id not in store:
            store[session_id] = []

        # Retrieve conversation history
        chat_history = store[session_id]

        # Execute Advanced RAG pipeline
        result = rag_chain.invoke({
            "input": question,
            "chat_history": chat_history[-10:]
        })

        # Extract answer
        if isinstance(result, dict):
            answer = (
                result.get("answer")
                or result.get("output")
                or result.get("response")
                or str(result)
            )
        else:
            answer = str(result)

        # Store user message
        chat_history.append(
            HumanMessage(content=question)
        )

        # Store AI response
        chat_history.append(
            AIMessage(content=answer)
        )

        return ChatResponse(
            answer=answer,
            session_id=session_id
        )

    except HTTPException:
        raise

    except Exception as e:

        print("RAG Error:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Error processing your question"
        )


# =========================================================
# 9. CLEAR SESSION HISTORY
# =========================================================

@app.delete(
    "/api/rag/session/{session_id}",
    dependencies=[Depends(verify_api_key)]
)
def clear_session(session_id: str):

    store.pop(session_id, None)

    return {
        "message": "Chat history cleared successfully",
        "session_id": session_id
    }
