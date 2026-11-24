
from langchain_core.documents import Document

from src.config import embeddings
from src.supabase_client import supabase


def retrieve_from_supabase(question, role=None, k=5):

    query_embedding = embeddings.embed_query(question)

    response = supabase.rpc(
        "match_document_chunks",
        {
            "query_embedding": query_embedding,
            "match_count": k,
            "filter_role": role
        }
    ).execute()

    documents = []

    for item in response.data:
        documents.append(
            Document(
                page_content=item["content"],
                metadata={
                    "document_name": item["document_name"],
                    "role": item["role"],
                    "similarity": item["similarity"]
                }
            )
        )

    return documents
