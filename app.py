# app.py — Legal AI Assistant with multi-category query router and chat memory

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
# Hardcoded for PoC only. Never do this in production.
CREDENTIALS = {
    "admin": "legal123",
}

def check_login(username, password):
    return CREDENTIALS.get(username) == password


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
Only add a review disclaimer at the end.
When a case file contains the word URGENT or mentions days remaining
to a limitation period, always lead your answer with that information
before anything else."""


# --- LOGIN GATE ---
# Renders the login form and blocks the rest of the app until authenticated.
# session_state.logged_in persists for the duration of the browser session.
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
# Searches laws, SOPs, and client files independently and combines results.
# Each category gets its own top_k allocation so no category can crowd
# out another regardless of how many chunks each contains.
# Laws get more slots (10) since the legal corpus is larger than SOPs/clients.
class MultiCategoryRetriever:
    def __init__(self, index):
        self.retrievers = {
            "law": VectorIndexRetriever(
                index=index,
                similarity_top_k=10,
                filters=MetadataFilters(filters=[
                    ExactMatchFilter(key="category", value="law")
                ])
            ),
            "sop": VectorIndexRetriever(
                index=index,
                similarity_top_k=5,
                filters=MetadataFilters(filters=[
                    ExactMatchFilter(key="category", value="sop")
                ])
            ),
            "client": VectorIndexRetriever(
                index=index,
                similarity_top_k=5,
                filters=MetadataFilters(filters=[
                    ExactMatchFilter(key="category", value="client")
                ])
            ),
        }

    def retrieve(self, query_str):
        # Query all three categories and combine.
        # Total chunks per query: up to 20 (10 law + 5 sop + 5 client).
        all_nodes = []
        for category, retriever in self.retrievers.items():
            try:
                nodes = retriever.retrieve(query_str)
                all_nodes.extend(nodes)
            except Exception as e:
                print(f"⚠️  Retriever failed for {category}: {e}")
        return all_nodes


# --- INDEX LOADER ---
# Loads ChromaDB index once and caches for the session.
# All conversations share the same index.
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


# --- QUERY ENGINE FACTORY ---
# Creates a RetrieverQueryEngine using the MultiCategoryRetriever.
# Uses tree_summarize for a single LLM call per query — faster and
# more reliable than refine mode on a small model.
# Each conversation gets its own engine instance so they stay independent.
def create_query_engine(index):
    llm = Ollama(
        model="llama3.2:3b",
        request_timeout=180.0,
        context_window=3072,
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


# --- CHAT MEMORY ---
# Injects the last 2 exchanges (4 messages) into each prompt.
# Kept at 2 exchanges to avoid injecting too much history into the
# small context window — enough for follow-up questions without
# polluting the context with stale information.
# Each message is truncated to 300 characters to protect the
# context window from very long previous answers.
def build_prompt_with_history(prompt, messages, max_exchanges=2):
    # Take only the last N exchanges from history
    recent = messages[-(max_exchanges * 2):]

    if not recent:
        return prompt

    history_str = ""
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        # Truncate long messages to avoid consuming the context window
        if len(content) > 300:
            content = content[:300] + "..."
        history_str += f"{role}: {content}\n"

    return f"""Previous conversation:
{history_str}
Current question: {prompt}"""


# --- NEW CONVERSATION ---
# Creates a fresh conversation with its own query engine and empty message list.
# UUID ensures each conversation has a unique identifier.
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


# --- SESSION STATE INITIALISATION ---
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
    st.caption(f"Signed in as **{st.session_state.username}**")

    if st.button("Sign out", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    if st.button("+ New chat", use_container_width=True):
        new_conversation(index)
        st.rerun()

    st.divider()

    # List conversations newest first
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

# Replay message history for this conversation
for msg in active_conv["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"📎 Sources: {', '.join(msg['sources'])}")

# Chat input
if prompt := st.chat_input("Ask a question about firm procedures or Ontario law..."):

    # Auto-name the conversation from the first message
    if len(active_conv["messages"]) == 0:
        active_conv["name"] = (
            prompt[:35] + "..." if len(prompt) > 35 else prompt
        )

    active_conv["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating response..."):

            # Inject last 2 exchanges of history into the prompt.
            # Excludes the message just appended so we don't inject
            # the current question as its own history.
            augmented_prompt = build_prompt_with_history(
                prompt,
                active_conv["messages"][:-1],
                max_exchanges=2
            )

            response = query_engine.query(augmented_prompt)
            answer = str(response)

            # Extract unique source filenames from retrieved chunks
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