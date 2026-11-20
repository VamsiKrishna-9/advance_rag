# Advanced RAG: Streamlit + FastAPI

This keeps the existing Streamlit app and CLI while adding a separate FastAPI
REST API. The existing `src/rag_pipeline.py` is reused.

## 1. Install

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env
```

Add a valid `GROQ_API_KEY` to `.env`. Keep `.env` private.

## 2. Confirm the FAISS index exists

The `vectorstore/` directory must contain both `index.faiss` and `index.pkl`.
The supplied project archive already included these files. Keep the index
files in the deployment repository if the index is static. Re-run `python
ingest.py` whenever you intentionally rebuild the index.

## 3. Run Streamlit

```bash
streamlit run app.py
```

## 4. Run FastAPI (in a second terminal)

```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs.

## 5. Chat API

`POST /api/rag/chat`

```json
{
  "question": "Ask a question grounded in the indexed documents",
  "session_id": "test-session-001"
}
```

Response:

```json
{
  "answer": "Generated answer",
  "session_id": "test-session-001"
}
```

Send the same `session_id` for follow-up questions. Different session IDs
have separate conversation histories. `DELETE /api/rag/session/{session_id}`
clears one session.

If `RAG_API_KEY` is configured, protected routes require:
`Authorization: Bearer YOUR_RAG_API_KEY`.

## 6. Railway deployment

Push this project to GitHub without `.env` or secrets. Create two Railway
services from the same repository:

- Streamlit service start command:
  `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`
- FastAPI service start command:
  `uvicorn api:app --host 0.0.0.0 --port $PORT`

Add `GROQ_API_KEY`, `RAG_API_KEY` (recommended), and `CORS_ORIGINS` to each
service that needs them. Generate a public domain for each service.

The current FAISS index is static and small, so it can be deployed with the
repository. A Railway Volume is only needed if you want to persist index
files that are created or modified at runtime. If using a Volume, copy the
index files into its mounted directory before starting the service.

## Important limitations

- Session history is held in process memory and is lost when the API restarts.
- For multiple API replicas or durable sessions, replace the in-memory
  session manager with Redis or a database.
- The current RAG pipeline returns an answer string, not source metadata;
  this API therefore does not claim to return citations or source documents.
- `health` confirms the application process is running. It does not independently
  verify a live Groq request.
