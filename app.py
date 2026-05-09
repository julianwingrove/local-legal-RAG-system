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
Never say a document is not in your context if it appears in your sources.
If a procedure is in your sources, reproduce its steps in full.
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
# Searches laws, SOPs, and client files independently and combines results.
# Each category gets top_k_per_category slots — no category can crowd out another.
# This means a cross-category question always gets relevant chunks from all three.
class MultiCategoryRetriever:
    def __init__(self, index, top_k_per_category=10):
        self.retrievers = {
            "law": VectorIndexRetriever(
                index=index,
                similarity_top_k=top_k_per_category,
                filters=MetadataFilters(filters=[
                    ExactMatchFilter(key="category", value="law")
                ])
            ),
            "sop": VectorIndexRetriever(
                index=index,
                similarity_top_k=top_k_per_category,
                filters=MetadataFilters(filters=[
                    ExactMatchFilter(key="category", value="sop")
                ])
            ),
            "client": VectorIndexRetriever(
                index=index,
                similarity_top_k=top_k_per_category,
                filters=MetadataFilters(filters=[
                    ExactMatchFilter(key="category", value="client")
                ])
            ),
        }

    def retrieve(self, query_str):
        # Query all three categories and combine results.
        # 5 chunks per category = 15 total chunks sent to the LLM.
        all_nodes = []
        for category, retriever in self.retrievers.items():
            try:
                nodes = retriever.retrieve(query_str)
                all_nodes.extend(nodes)
            except Exception as e:
                print(f"⚠️  Retriever failed for {category}: {e}")
        return all_nodes


# --- INDEX LOADER ---
# Loads ChromaDB index once and caches it for the session.
# All conversations share the same index — only the query engines differ.
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
# Each conversation gets its own engine instance so they stay independent.
# We use tree_summarize for a single LLM call per query — faster and
# more memory-efficient than the default refine mode on 8GB.
def create_query_engine(index):
    llm = Ollama(
        model="llama3.2:3b",
        request_timeout=180.0,
        context_window=3072,
        keep_alive="60m",
        system_prompt=SYSTEM_PROMPT
    )

    retriever = MultiCategoryRetriever(index, top_k_per_category=5)

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        response_synthesizer=get_response_synthesizer(
            llm=llm,
            response_mode="tree_summarize"
        )
    )


# --- CHAT MEMORY ---
# Instead of relying on the chat engine to manage memory,
# we inject the last few exchanges directly into each prompt.
# This is more reliable with small models because it avoids
# the condensation step that confuses llama3.2:3b.
# We keep the last 3 exchanges (6 messages) to stay within
# the 3072 token context window comfortably.
def build_prompt_with_history(prompt, messages, max_exchanges=3):
    # Grab the last N user/assistant pairs (excluding the current message)
    recent = messages[-(max_exchanges * 2):]

    if not recent:
        return prompt

    history_str = ""
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Truncate long messages to avoid blowing the context window
        content = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
        history_str += f"{role}: {content}\n"

    return f"""Previous conversation:
{history_str}
Current question: {prompt}"""


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
    st.error(f"Failed to load index. Have you run `python ingest.py` first?\n\nError: {e}")
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

# Replay message history for this conversation
for msg in active_conv["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"📎 Sources: {', '.join(msg['sources'])}")

# Chat input
if prompt := st.chat_input("Ask a question about firm procedures or Ontario law..."):

    # Auto-name conversation from first message
    if len(active_conv["messages"]) == 0:
        active_conv["name"] = prompt[:35] + "..." if len(prompt) > 35 else prompt

    active_conv["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating response..."):

            # Inject recent chat history into the prompt before querying.
            # This gives the model memory of the conversation without
            # relying on the chat engine's condensation step.
            augmented_prompt = build_prompt_with_history(
                prompt,
                active_conv["messages"][:-1]  # exclude the message just appended
            )

            response = query_engine.query(augmented_prompt)
            answer = str(response)

            # Extract source filenames from retrieved nodes
            sources = list({
                node.metadata.get("file_name", "Unknown")
                for node in response.source_nodes
            })

            st.markdown(answer)
            if sources:
                st.caption(f"📎 Sources: {', '.join(sources)}")
            st.caption("⚠️ AI-assisted research only. Must be reviewed by a licensed lawyer.")

    active_conv["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })