# app.py — Legal AI Assistant with multi-conversation sidebar

import uuid
import streamlit as st
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# --- CREDENTIALS ---
# Hardcoded for PoC only. Never do this in production.
CREDENTIALS = {
    "admin": "legal123",
}

def check_login(username, password):
    return CREDENTIALS.get(username) == password

# layout="wide" gives more horizontal space for the sidebar + chat layout
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide"
)

# --- LOGIN GATE ---
# If not logged in, show the login screen and stop the rest of the app
# from rendering. Once authenticated, session_state.logged_in stays
# True for the duration of the session so the gate is skipped on re-runs.
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

SYSTEM_PROMPT = """You are a legal research assistant for a law firm.
Answer questions using ONLY the documents provided in your context.
Always cite which document your answer comes from, including the section number.
If the answer is not in your documents, say so clearly — do not guess.
When documents contain step-by-step procedures or checklists, reproduce
them fully and accurately — this is not legal advice, it is procedural guidance.
Never say "I can't provide instructions" — if the answer is in your documents,
provide it directly and completely.
Only add a review disclaimer at the end."""


# --- INDEX LOADER ---
# Load the ChromaDB index once and cache it — shared across all conversations.
# We separate this from the chat engine so each conversation can have its own
# engine instance (with its own memory) while sharing the same document index.
@st.cache_resource
def load_index():
    embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection("legal_docs")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )


# --- CHAT ENGINE FACTORY ---
# Creates a fresh chat engine with its own memory buffer.
# Called once per new conversation so each conversation tracks
# its own context independently from all others.
def create_chat_engine(index):
    llm = Ollama(
        model="llama3.2:3b",
        request_timeout=180.0,
        context_window=3072,
        keep_alive="60m",
        system_prompt=SYSTEM_PROMPT
    )
    return index.as_chat_engine(
        llm=llm,
        chat_mode="context",
        similarity_top_k=10,
        system_prompt=SYSTEM_PROMPT
    )


# --- NEW CONVERSATION ---
# Creates a new conversation entry in session_state with:
#   - a unique ID (uuid)
#   - a default name that updates to the first message after sending
#   - an empty message list for display
#   - its own fresh chat engine instance with isolated memory
def new_conversation(index):
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {
        "name": "New chat",
        "messages": [],
        "engine": create_chat_engine(index)
    }
    st.session_state.active_conv_id = conv_id


# --- LOAD INDEX ---
try:
    index = load_index()
except Exception as e:
    st.error(f"Failed to load index. Have you run `python ingest.py` first?\n\nError: {e}")
    st.stop()


# --- SESSION STATE INITIALISATION ---
# Initialise conversations dict and active conversation on first run.
# On subsequent Streamlit re-runs these already exist and are skipped.
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "active_conv_id" not in st.session_state:
    st.session_state.active_conv_id = None

# Always ensure at least one conversation exists
if not st.session_state.conversations:
    new_conversation(index)


# --- SIDEBAR ---
with st.sidebar:
    st.title("⚖️ Legal AI")
    st.caption("🔒 Fully local")

    # Show who is logged in and a logout button
    st.caption(f"Signed in as **{st.session_state.username}**")
    if st.button("Sign out", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    # New chat button — creates a fresh conversation and switches to it
    if st.button("+ New chat", use_container_width=True):
        new_conversation(index)
        st.rerun()

    st.divider()

    # List all conversations — most recent at the top
    # Each conversation gets a select button and a delete button side by side
    for conv_id in reversed(list(st.session_state.conversations.keys())):
        conv = st.session_state.conversations[conv_id]
        is_active = conv_id == st.session_state.active_conv_id

        # Two columns: conversation name button + delete button
        col1, col2 = st.columns([5, 1])

        with col1:
            # Highlighted differently if this is the active conversation
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
                # If no conversations left, create a fresh one automatically
                if not st.session_state.conversations:
                    new_conversation(index)
                # If we deleted the active one, switch to the most recent remaining
                elif st.session_state.active_conv_id == conv_id:
                    st.session_state.active_conv_id = list(
                        st.session_state.conversations.keys()
                    )[-1]
                st.rerun()


# --- MAIN CHAT AREA ---
st.title("⚖️ Legal AI Assistant")
st.caption("🔒 Fully local — no data leaves this machine")

# Get the active conversation and its engine
active_conv = st.session_state.conversations[st.session_state.active_conv_id]
query_engine = active_conv["engine"]

# Replay message history for this conversation
for msg in active_conv["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"📎 Sources: {', '.join(msg['sources'])}")

# Chat input
if prompt := st.chat_input("Ask a question about firm procedures or Ontario law..."):

    # Auto-name the conversation from the first message, truncated to 35 chars
    if len(active_conv["messages"]) == 0:
        active_conv["name"] = prompt[:35] + "..." if len(prompt) > 35 else prompt

    active_conv["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating response..."):
            response = query_engine.chat(prompt)
            answer = str(response)

            # Extract source filenames from chat engine response
            sources = []
            try:
                for source in response.sources:
                    for node in source.raw_output.source_nodes:
                        fname = node.metadata.get("file_name", "Unknown")
                        if fname not in sources:
                            sources.append(fname)
            except Exception:
                sources = []

            st.markdown(answer)
            if sources:
                st.caption(f"📎 Sources: {', '.join(sources)}")
            st.caption("⚠️ AI-assisted research only. Must be reviewed by a licensed lawyer.")

    active_conv["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })