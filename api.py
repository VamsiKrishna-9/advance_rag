
import os
import hmac
from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import (
    HTTPBearer,
    HTTPAuthorizationCredentials,
)

from src.rag_pipeline import rag_chain
from src.api_schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
)
from src.session_manager import session_manager


# =========================================================
# 1. ENVIRONMENT CONFIGURATION
# =========================================================

load_dotenv(override=True)

RAG_API_KEY = os.getenv("RAG_API_KEY")

# =========================================================
# 2. SWAGGER BEARER AUTHENTICATION
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
# 3. FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Advanced RAG API",
    description="REST API for the existing LangChain Advanced RAG pipeline.",
    version="1.0.0"
)


# =========================================================
# 4. CORS CONFIGURATION
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
# 5. ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():
    return {
        "service": "Advanced RAG API",
        "status": "running",
        "docs": "/docs"
    }


# =========================================================
# 6. HEALTH CHECK
# =========================================================

@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "healthy",
        "service": "advanced-rag-api",
        "rag_loaded": rag_chain is not None
    }


# =========================================================
# 7. CHAT ENDPOINT
# =========================================================

@app.post(
    "/api/rag/chat",
    response_model=ChatResponse,
    dependencies=[Depends(verify_api_key)]
)
def chat(request: ChatRequest):

    try:
        # Retrieve previous conversation history
        chat_history = session_manager.get_history(
            request.session_id
        )

        # Execute Advanced RAG pipeline
        result = rag_chain.invoke(
            {
                "input": request.question,
                "chat_history": chat_history
            }
        )

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

        # Save conversation history
        session_manager.append_turn(
            request.session_id,
            request.question,
            answer
        )

        return ChatResponse(
            answer=answer,
            session_id=request.session_id
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG processing failed: {str(e)}"
        )


# =========================================================
# 8. CLEAR CONVERSATION SESSION
# =========================================================

@app.delete(
    "/api/rag/session/{session_id}",
    dependencies=[Depends(verify_api_key)]
)
def clear_session(session_id: str):

    session_manager.clear(session_id)

    return {
        "message": "Conversation history cleared successfully",
        "session_id": session_id
    }
