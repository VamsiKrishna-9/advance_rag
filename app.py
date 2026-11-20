import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.rag_pipeline import rag_chain

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="MediAI | RAG Chat",
    page_icon="🏥",
    layout="centered",
)

st.title("🏥 MediAI Assistant")
st.caption("Ask questions about the MediAI knowledge base.")
st.divider()

# --------------------------------------------------
# Per-browser-session conversation state
# Keep LangChain messages separate from display messages.
# --------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

with st.sidebar:
    st.subheader("Conversation")
    st.caption("Ask a question, then use follow-ups to continue the conversation.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.display_messages = []
        st.rerun()

# Render previous turns
for message in st.session_state.display_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------
# Question -> existing RAG chain -> answer
# The same intent routing, contextualization, retrieval,
# prompts, LLM, and FAISS retriever from src/ are reused.
# --------------------------------------------------
question = st.chat_input("Ask a question about the documents...")

if question:
    st.session_state.display_messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = rag_chain.invoke({
                    "input": question,
                    "chat_history": st.session_state.chat_history[-10:],
                })
                if not isinstance(answer, str):
                    answer = str(answer)
                st.markdown(answer)
            except Exception as exc:
                st.error(
                    "I couldn't process that question. Check that your "
                    "environment, dependencies, Groq API key, and FAISS "
                    "vectorstore are configured correctly."
                )
                st.exception(exc)
                answer = None

    if answer is not None:
        st.session_state.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer),
        ])
        st.session_state.display_messages.append(
            {"role": "assistant", "content": answer}
        )
