
from langchain_core.runnables import RunnableLambda

from src.supabase_retriever import retrieve_from_supabase

from src.chains import (
    contextualize_chain,
    document_answer_chain,
    intent_chain,
    history_answer_chain,
    general_chain
)


def retrieve_documents(inputs):

    # 1. Rewrite the question
    standalone_question = contextualize_chain.invoke(
        {
            "input": inputs["input"],
            "chat_history": inputs["chat_history"]
        }
    )

    print("\nRewritten Question:", standalone_question)

    # 2. Retrieve relevant chunks from Supabase
    documents = retrieve_from_supabase(
        standalone_question,
        role=inputs.get("role"),
        k=5
    )

    # 3. Combine document chunks
    context = "\n\n".join(
        [doc.page_content for doc in documents]
    )

    # 4. Handle empty retrieval
    if not context.strip():
        return "I don't know based on the provided documents."

    # 5. Generate document-based answer
    answer = document_answer_chain.invoke(
        {
            "input": standalone_question,
            "context": context
        }
    )

    return answer


def route_request(inputs):

    question = inputs["input"]
    history = inputs["chat_history"]

    # Identify intent
    intent = intent_chain.invoke(
        {
            "input": question,
            "chat_history": history
        }
    ).strip().upper()

    print("\nDetected Intent:", intent)

    if intent == "DOCUMENT":
        return retrieve_documents(inputs)

    elif intent == "HISTORY":
        return history_answer_chain.invoke(
            {
                "input": question,
                "chat_history": history
            }
        )

    elif intent == "GENERAL":
        return general_chain.invoke(
            {
                "input": question,
                "chat_history": history
            }
        )

    else:
        return general_chain.invoke(
            {
                "input": question,
                "chat_history": history
            }
        )


rag_chain = RunnableLambda(route_request)
