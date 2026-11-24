
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import embeddings
from src.supabase_client import supabase


# 1. PDF location
PDF_PATH = Path("data/MediAI_Complete_Hospital_Knowledge_Base.pdf")

# 2. Load PDF
loader = PyPDFLoader(str(PDF_PATH))
documents = loader.load()

print("Total pages:", len(documents))


# 3. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))


# 4. Generate embeddings
texts = [chunk.page_content for chunk in chunks]

vectors = embeddings.embed_documents(texts)


# 5. Prepare Supabase records
records = []

for chunk, vector in zip(chunks, vectors):

    records.append({
        "content": chunk.page_content,
        "embedding": vector,
        "document_name": PDF_PATH.name,
        "role": "all",
        "metadata": {
            **chunk.metadata,
            "source": PDF_PATH.name
        }
    })


# 6. Upload to Supabase in batches
for i in range(0, len(records), 100):

    batch = records[i:i + 100]

    supabase.table("document_chunks").insert(
        batch
    ).execute()

    print(f"Uploaded {min(i + 100, len(records))}/{len(records)}")


print("Successfully stored all document chunks in Supabase!")
