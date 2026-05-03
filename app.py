# app.py — The chat interface

import streamlit as st
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

st.set_page_config(page_title="Legal AI Assistant", page_icon="⚖️")
st.title("⚖️ Legal AI Assistant")
st.caption("🔒 Fully local — no data leaves this machine")

SYSTEM_PROMPT = """You are a legal research assistant for a law firm. 
Answer questions using ONLY the documents provided in your context.
Always cite which document your answer comes from.
If the answer is not in your documents, say so clearly — do not guess.
Never provide a final legal opinion; always note that answers require 
review by a licensed lawyer."""

@st.cache_resource
def load_engine():
    llm = Ollama(
        model="llama3.2:3b",
        request_timeout=180.0,
        system_prompt=SYSTEM_PROMPT
    )
    embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection("legal_docs")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=embed_model
    )
    return index.as_query_engine(llm=llm, similarity_top_k=5)

# Load the engine
try:
    query_engine = load_engine()
except Exception as e:
    st.error(f"Failed to load engine. Have you run `python ingest.py` first? \n\nError: {e}")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"📎 Sources: {', '.join(msg['sources'])}")

# Input
if prompt := st.chat_input("Ask a question about firm procedures or Ontario law..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating response..."):
            response = query_engine.query(prompt)
            answer = str(response)

            # Extract source filenames
            sources = list({
                node.metadata.get("file_name", "Unknown")
                for node in response.source_nodes
            })

            st.markdown(answer)
            if sources:
                st.caption(f"📎 Sources: {', '.join(sources)}")
            st.caption("⚠️ AI-assisted research only. Must be reviewed by a licensed lawyer.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })