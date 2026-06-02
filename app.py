# app.py — Legal AI Assistant
# 3-document focused corpus: Margaret Chen, Limitations Act, SOP-002
# Embedding: nomic-embed-text | Chunk size: 150 | Context window: 6500

import uuid
import streamlit as st
import chromadb

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide"
)


# --- CREDENTIALS ---
CREDENTIALS = {
    "admin": "legal123",
}

def check_login(username, password):
    return CREDENTIALS.get(username) == password


# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are a legal research assistant for a law firm.
Answer questions using ONLY the documents provided in your context.
Always cite which document your answer comes from, including the section number.
If the answer is not in your documents, say so clearly — do not guess.
When documents contain step-by-step procedures or checklists, reproduce
them fully and accurately — this is not legal advice, it is procedural guidance.
Never say "I can't provide instructions" — if the answer is in your documents,
provide it directly and completely.
Never say a document is not in your context if its filename appears in your sources.
Never reference people, cases, or documents not present in your current sources.
If a case file is in your sources, summarise it fully and accurately.
When a case file contains the word URGENT or mentions days remaining
to a limitation period, always lead your answer with that information
before anything else.
Only add a review disclaimer at the end."""


# --- LOGIN GATE ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("⚖️ Legal AI Assistant")
    st.subheader("Sign in to continue")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        if check_login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Incorrect username or password.")

    st.stop()


# --- MULTI-CATEGORY RETRIEVER ---
# Even with 3 documents, the Limitations Act PDF contains hundreds of
# chunks that would dominate a single retriever. Per-category allocation
# guarantees the SOP and client file always reach the LLM.
#
# Token budget: 30 chunks × 150 tokens = 4,500 tokens of content
# + 170 system/question + 1,474 answer headroom = 6,500 context window
class MultiCategoryRetriever:
    def __init__(self, index):
        self.retrievers = {
            # Law: Limitations Act PDF is large — 12 slots retrieves
            # the most relevant legislative provisions per query.
            "law": VectorIndexRetriever(
                index=index,
                similarity_top_k=20,
                filters=MetadataFilters(filters=[
                    ExactMatchFilter(key="category", value="law")
                ])
            ),
            # SOP: SOP-002 is a medium document — 10 slots covers
            # nearly the entire document, ensuring no procedure is missed.
            "sop": VectorIndexRetriever(
                index=index,
                similarity_top_k=18,
                filters=MetadataFilters(filters=[
                    ExactMatchFilter(key="category", value="sop")
                ])
            ),
            # Client: Margaret Chen's file is small — 8 slots covers
            # virtually the entire case file on every query.
            "client": VectorIndexRetriever(
                index=index,
                similarity_top_k=14,
                filters=MetadataFilters(filters=[
                    ExactMatchFilter(key="category", value="client")
                ])
            ),
        }

    def retrieve(self, query_str):
        # Query all three categories simultaneously and combine.
        # Total: up to 30 chunks (12 law + 10 sop + 8 client).
        all_nodes = []
        for category, retriever in self.retrievers.items():
            try:
                nodes = retriever.retrieve(query_str)
                all_nodes.extend(nodes)
            except Exception as e:
                print(f"⚠️  Retriever failed for {category}: {e}")
        return all_nodes


# --- INDEX LOADER ---
@st.cache_resource
def load_index():
    # Must match the embedding model used in ingest.py.
    embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection("legal_docs")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )


# --- QUERY ENGINE FACTORY ---
# context_window=6500: optimal ceiling for 8GB unified memory.
# Fits 30 chunks × 150 tokens = 4,500 tokens of content plus
# system prompt, question, and ~1,474 tokens of answer headroom.
# tree_summarize: single LLM call — faster and more reliable
# than refine mode on a 3B model.
# Chat history disabled: entire context window reserved for
# document content, maximising what the LLM can see per query.
def create_query_engine(index):
    llm = Ollama(
        model="llama3.2:3b",
        request_timeout=180.0,
        context_window=6500,
        keep_alive="60m",
        system_prompt=SYSTEM_PROMPT
    )

    retriever = MultiCategoryRetriever(index)

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        response_synthesizer=get_response_synthesizer(
            llm=llm,
            response_mode="tree_summarize"
        )
    )


# --- NEW CONVERSATION ---
def new_conversation(index):
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {
        "name": "New chat",
        "messages": [],
        "engine": create_query_engine(index)
    }
    st.session_state.active_conv_id = conv_id


# --- LOAD INDEX ---
try:
    index = load_index()
except Exception as e:
    st.error(
        f"Failed to load index. Have you run `python ingest.py` first?\n\nError: {e}"
    )
    st.stop()


# --- SESSION STATE ---
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "active_conv_id" not in st.session_state:
    st.session_state.active_conv_id = None

if not st.session_state.conversations:
    new_conversation(index)


# --- SIDEBAR ---
with st.sidebar:
    st.title("⚖️ Legal AI")
    st.caption("🔒 Fully local")
    st.caption(f"Signed in as **{st.session_state.username}**")

    if st.button("Sign out", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    if st.button("+ New chat", use_container_width=True):
        new_conversation(index)
        st.rerun()

    st.divider()

    for conv_id in reversed(list(st.session_state.conversations.keys())):
        conv = st.session_state.conversations[conv_id]
        is_active = conv_id == st.session_state.active_conv_id

        col1, col2 = st.columns([5, 1])

        with col1:
            if st.button(
                conv["name"],
                key=f"select_{conv_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.active_conv_id = conv_id
                st.rerun()

        with col2:
            if st.button("✕", key=f"delete_{conv_id}", use_container_width=True):
                del st.session_state.conversations[conv_id]
                if not st.session_state.conversations:
                    new_conversation(index)
                elif st.session_state.active_conv_id == conv_id:
                    st.session_state.active_conv_id = list(
                        st.session_state.conversations.keys()
                    )[-1]
                st.rerun()


# --- MAIN CHAT AREA ---
st.title("⚖️ Legal AI Assistant")
st.caption("🔒 Fully local — no data leaves this machine")

active_conv = st.session_state.conversations[st.session_state.active_conv_id]
query_engine = active_conv["engine"]

for msg in active_conv["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"📎 Sources: {', '.join(msg['sources'])}")

if prompt := st.chat_input("Ask a question about firm procedures or Ontario law..."):

    if len(active_conv["messages"]) == 0:
        active_conv["name"] = (
            prompt[:35] + "..." if len(prompt) > 35 else prompt
        )

    active_conv["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating response..."):

            # Direct query — no history injection.
            # Full 6,500 token context window available for document content.
            response = query_engine.query(prompt)
            answer = str(response)

            sources = list({
                node.metadata.get("file_name", "Unknown")
                for node in response.source_nodes
            })

            st.markdown(answer)
            if sources:
                st.caption(f"📎 Sources: {', '.join(sources)}")
            st.caption(
                "⚠️ AI-assisted research only. "
                "Must be reviewed by a licensed lawyer."
            )

    active_conv["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })