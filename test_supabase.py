
from src.supabase_retriever import retrieve_from_supabase

question = "What services are available in MediAI hospital?"

documents = retrieve_from_supabase(
    question,
    role="all",
    k=5
)

print("\nRetrieved Documents:", len(documents))

for i, doc in enumerate(documents, 1):
    print(f"\n--- Result {i} ---")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
