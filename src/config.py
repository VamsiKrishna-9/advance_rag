import os
from pathlib import Path
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Resolve paths relative to the project root so Streamlit and FastAPI
# can be started from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Initial Embeddings
embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# Load FAISS vector database

vectorstore=FAISS.load_local(
    str(PROJECT_ROOT / "vectorstore"),
    embeddings,
    allow_dangerous_deserialization=True
)

# Create the Retriver
retriever=vectorstore.as_retriever(
    search_kwargs={"k":3}
)

# Initialize The LLM
llm=ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)