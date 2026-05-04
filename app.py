# app.py — The chat interface
# This script runs the Streamlit web app that the lawyer interacts with.
# It loads the indexed documents from ChromaDB and uses the local LLM
# to answer questions based on what was indexed by ingest.py.

# --- IMPORTS ---

import streamlit as st  # Streamlit turns this Python script into a web UI automatically.
                        # Every time the user does something (sends a message, etc.),
                        # Streamlit re-runs the entire script from top to bottom.

import chromadb  # The local vector database where your indexed documents live.

# VectorStoreIndex: the LlamaIndex object that wraps your ChromaDB index
#                   and knows how to search it.
# StorageContext: tells LlamaIndex where data is stored (in our case, ChromaDB).
from llama_index.core import VectorStoreIndex, StorageContext

# Ollama: the connector that lets LlamaIndex talk to your locally running LLM.
from llama_index.llms.ollama import Ollama

# OllamaEmbedding: the connector for the local embedding model (nomic-embed-text).
# Must be the same model used in ingest.py — mixing models breaks similarity search.
from llama_index.embeddings.ollama import OllamaEmbedding

# ChromaVectorStore: the bridge/adapter between LlamaIndex and ChromaDB.
from llama_index.vector_stores.chroma import ChromaVectorStore


# --- PAGE CONFIGURATION ---
# These must be the first Streamlit calls in the script.
# page_title sets the browser tab title, page_icon sets the favicon.
st.set_page_config(page_title="Legal AI Assistant", page_icon="⚖️")
st.title("⚖️ Legal AI Assistant")
st.caption("🔒 Fully local — no data leaves this machine")


# --- SYSTEM PROMPT ---
# This is the instruction set given to the LLM before every conversation.
# It defines the AI's role, constraints, and behaviour.
# This is one of the most important parts of the app — a well-written system
# prompt is what prevents the model from hallucinating or going off-script.
# Key rules enforced here:
#   - Only answer from the provided documents (no general knowledge)
#   - Always cite sources
#   - Admit when it doesn't know rather than guessing
#   - Never act as a lawyer or give final legal opinions
SYSTEM_PROMPT = """You are a legal research assistant for a law firm.
Answer questions using ONLY the documents provided in your context.
Always cite which document your answer comes from, including the section number.
If the answer is not in your documents, say so clearly — do not guess.
When documents contain step-by-step procedures or checklists, reproduce
them fully and accurately — this is not legal advice, it is procedural guidance.
Never say "I can't provide instructions" — if the answer is in your documents,
provide it directly and completely.
Only add a review disclaimer at the end."""


# --- ENGINE LOADER ---
# @st.cache_resource is a Streamlit decorator that runs this function ONCE
# and then caches (stores) the result in memory for the entire session.
# Without it, Streamlit would reload the LLM and reconnect to ChromaDB on
# every single message — adding 10-15 seconds of delay each time.
# This is why the first message is slower: the engine is loading.
# Every message after that reuses the already-loaded engine instantly.
@st.cache_resource
def load_engine():
    # Initialise the local LLM via Ollama.
    # model: which model to use (must already be pulled via `ollama pull`)
    # request_timeout: how many seconds to wait before giving up on a response.
    #                  180s is generous — reduce to 60s once you're happy with speed.
    # system_prompt: the instruction set defined above, sent before every query.
    llm = Ollama(
        model="llama3.2:3b",
        request_timeout=180.0,
        context_window=4096,
        keep_alive="60m",
        system_prompt=SYSTEM_PROMPT
    )

    # Initialise the same embedding model used during ingestion.
    # This is critical: the query must be embedded with the same model
    # that was used to embed the documents, otherwise the vectors won't
    # be comparable and similarity search will return garbage results.
    embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    # Connect to the ChromaDB database on disk (created by ingest.py).
    # PersistentClient reads from the ./chroma_db folder — it does NOT
    # load all vectors into memory. It queries the database file on demand.
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # Connect to the specific collection created during ingestion.
    # get_or_create_collection: if "legal_docs" exists, connect to it.
    # If it doesn't exist (i.e. ingest.py hasn't been run yet), it creates
    # an empty one — which is why queries would return no results, not an error.
    collection = chroma_client.get_or_create_collection("legal_docs")

    # Wrap the ChromaDB collection in LlamaIndex's adapter.
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # Tell LlamaIndex to use ChromaDB as its storage backend.
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load the existing index from ChromaDB — this does NOT re-embed anything.
    # It simply points LlamaIndex at the already-computed vectors in ChromaDB
    # so it can search them. This is much faster than from_documents() in ingest.py.
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=embed_model
    )

    # Create the query engine — the object that handles the full RAG pipeline:
    #   1. Embed the user's question using embed_model
    #   2. Search ChromaDB for the top 3 most similar document chunks
    #   3. Send those chunks + the question to the LLM
    #   4. Return the LLM's answer along with which chunks it used (source_nodes)
    #
    # similarity_top_k=3: retrieve the 3 closest document chunks.
    #   Higher = more context but slower and more memory pressure.
    #   Lower = faster but may miss relevant information.
    #
    # response_mode="tree_summarize": calls the LLM once using all 3 chunks together.
    #   The alternative (default "compact_and_refine") calls the LLM once PER chunk,
    #   which is 3x slower. tree_summarize is the right choice for an 8GB machine.
    return index.as_chat_engine(
        llm=llm,
        chat_mode="context",
        similarity_top_k=3,
        system_prompt=SYSTEM_PROMPT
    )


# --- ENGINE INITIALISATION ---
# Attempt to load the engine when the app starts.
# If it fails (e.g. ChromaDB is empty because ingest.py hasn't been run,
# or Ollama isn't running), show a friendly error and stop the app
# rather than crashing with a raw Python traceback.
try:
    query_engine = load_engine()
except Exception as e:
    st.error(f"Failed to load engine. Have you run `python ingest.py` first? \n\nError: {e}")
    st.stop()  # Halts the rest of the script — nothing below this runs.


# --- CHAT HISTORY ---
# st.session_state is Streamlit's way of persisting data across re-runs.
# Because Streamlit re-runs the whole script on every interaction,
# regular Python variables would reset to empty on each message.
# session_state survives re-runs for the duration of the browser session.
#
# Here we initialise an empty list the first time the app loads.
# On every subsequent re-run, this block is skipped because "messages"
# already exists in session_state.
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- DISPLAY CHAT HISTORY ---
# On every re-run, replay all previous messages to the screen.
# This is what makes it look like a continuous conversation —
# Streamlit doesn't natively remember what's on screen, so we
# re-render the full history from session_state each time.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):  # "user" or "assistant" — controls the avatar
        st.markdown(msg["content"])
        # Only show the sources line if this message had sources attached.
        # User messages don't have sources; only assistant responses do.
        if "sources" in msg and msg["sources"]:
            st.caption(f"📎 Sources: {', '.join(msg['sources'])}")


# --- CHAT INPUT & RESPONSE ---
# st.chat_input renders the text box at the bottom of the screen.
# The := (walrus operator) both assigns the value AND checks if it's non-empty.
# This entire block only runs when the user has typed something and pressed Enter.
if prompt := st.chat_input("Ask a question about firm procedures or Ontario law..."):

    # Save the user's message to history so it persists on re-runs.
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display the user's message in the chat UI immediately.
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display the assistant's response in its own chat bubble.
    with st.chat_message("assistant"):
        # st.spinner shows a loading animation while the LLM is thinking.
        with st.spinner("Searching documents and generating response..."):

            # This is the core RAG call. It:
            #   1. Embeds `prompt` into a vector
            #   2. Searches ChromaDB for the 3 closest document chunks
            #   3. Sends those chunks + prompt to llama3.2:3b
            #   4. Returns the response object
            response = query_engine.chat(prompt)

            # Convert the response object to a plain string for display.
            answer = str(response)

            # Extract the filenames of the source documents that were retrieved.
            # response.source_nodes is a list of the chunks that were used.
            # Each node has a metadata dict containing info from ingestion —
            # including "file_name" which was automatically set by SimpleDirectoryReader.
            # We use a set comprehension (not a list) to deduplicate: if two chunks
            # came from the same file, we only want to show that filename once.
            # Then we convert to a list for joining into a comma-separated string.
            sources = []
            try:
                for source in response.sources:
                    for node in source.raw_output.source_nodes:
                        fname = node.metadata.get("file_name", "Unknown")
                        if fname not in sources:
                            sources.append(fname)
            except Exception:
                sources = []

            # Render the answer as markdown (supports bold, bullet points, etc.)
            st.markdown(answer)

            # Show which documents the answer was drawn from.
            if sources:
                st.caption(f"📎 Sources: {', '.join(sources)}")

            # Always show this disclaimer — it's a professional and ethical requirement.
            # No AI output should ever be treated as final legal advice.
            st.caption("⚠️ AI-assisted research only. Must be reviewed by a licensed lawyer.")

    # Save the assistant's response to history.
    # We store sources alongside the content so they can be re-displayed
    # when the chat history is replayed on the next Streamlit re-run.
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })