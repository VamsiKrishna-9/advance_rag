
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import embeddings
from src.supabase_client import supabase


BUCKET_NAME = "mediai-knowledge"

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)


def ingest_pdf(file_name, role):

    print(f"\nProcessing: {file_name}")

    # 1. Download PDF from Supabase Storage
    response = supabase.storage.from_(
        BUCKET_NAME
    ).download(file_name)

    # 2. Save temporarily
    pdf_path = TEMP_DIR / Path(file_name).name
    pdf_path.write_bytes(response)

    try:
        # 3. Extract PDF text
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        # 4. Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No text extracted from PDF")

        # 5. Generate embeddings
        texts = [chunk.page_content for chunk in chunks]

        vectors = embeddings.embed_documents(texts)

        # 6. Prepare database records
        records = []

        for chunk, vector in zip(chunks, vectors):

            records.append({
                "content": chunk.page_content,
                "embedding": vector,
                "document_name": Path(file_name).name,
                "role": role,
                "metadata": {
                    **chunk.metadata,
                    "source": file_name
                }
            })

        # 7. Insert in batches
        for i in range(0, len(records), 100):

            batch = records[i:i + 100]

            supabase.table(
                "document_chunks"
            ).insert(batch).execute()

        print(f"Successfully stored {len(records)} chunks")

    finally:
        # 8. Remove temporary PDF
        pdf_path.unlink(missing_ok=True)


if __name__ == "__main__":

    documents = [
        ("Patient_Knowledge.pdf", "patient"),
        ("Doctor_Knowledge.pdf", "doctor"),
        ("Staff_Knowledge.pdf", "staff"),
        ("Admin_Knowledge.pdf", "admin")
    ]

    for file_name, role in documents:
        ingest_pdf(file_name, role)
