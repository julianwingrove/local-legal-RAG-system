# ingest.py — Run this once to index your documents
# This script reads your documents, converts them into vector embeddings,
# and stores them in ChromaDB so the app can search them later.

# --- IMPORTS ---
# chromadb is the local vector database we use to store and search embeddings
import chromadb

import fitz  # pymupdf
import os
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# SimpleDirectoryReader: scans a folder and loads all documents into memory
# VectorStoreIndex: the main object that coordinates embedding + storage
# StorageContext: tells LlamaIndex *where* to store the vectors (in our case, ChromaDB)
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext

# OllamaEmbedding: uses the local nomic-embed-text model (via Ollama) to convert
# text into vectors. This runs 100% locally — no API calls, no internet.
from llama_index.embeddings.ollama import OllamaEmbedding

# ChromaVectorStore: the bridge/adapter between LlamaIndex and ChromaDB.
# LlamaIndex doesn't talk to ChromaDB directly — this class translates between them.
from llama_index.vector_stores.chroma import ChromaVectorStore


# --- STEP 1: LOAD DOCUMENTS ---
print("🔍 Loading documents...")

# SimpleDirectoryReader scans the ./documents folder and all its subfolders
# (recursive=True means it goes into /laws, /sops, /clients etc.)
# required_exts filters to only load .txt, .pdf, and .docx files — ignoring
# things like .DS_Store, .gitignore, or any other files in that folder.
# .load_data() reads the content of every matched file into memory as
# a list of Document objects — one per file (or per page for PDFs).
documents = []

for root, dirs, files in os.walk("./documents"):
    for filename in files:
        filepath = os.path.join(root, filename)
        ext = os.path.splitext(filename)[1].lower()

        try:
            if ext == ".pdf":
                # Use pymupdf for robust PDF parsing
                doc = fitz.open(filepath)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                if text.strip():
                    documents.append(Document(
                        text=text,
                        metadata={"file_name": filename, "file_path": filepath}
                    ))
                else:
                    print(f"⚠️  No text extracted from {filename} — skipping")

            elif ext in [".txt", ".docx"]:
                # Use SimpleDirectoryReader for non-PDF files
                from llama_index.core import SimpleDirectoryReader
                docs = SimpleDirectoryReader(
                    input_files=[filepath]
                ).load_data()
                for d in docs:
                    d.metadata["file_name"] = filename
                documents.extend(docs)

        except Exception as e:
            print(f"⚠️  Failed to load {filename}: {e}")

print(f"✅ Loaded {len(documents)} documents")


# --- STEP 2: SET UP THE VECTOR STORE ---
print("🔢 Setting up vector store...")

# Tell LlamaIndex to use nomic-embed-text (running locally via Ollama) as the
# embedding model. This is the model that converts text → vectors.
# Every document chunk and every future query will be embedded using this same model.
# IMPORTANT: you must always use the same embedding model for both ingestion
# and querying — mixing models will produce garbage search results.
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Create (or connect to) the ChromaDB database stored on disk at ./chroma_db.
# PersistentClient means the data survives after the script finishes —
# as opposed to an in-memory client that disappears when the process ends.
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Get or create a "collection" inside ChromaDB named "legal_docs".
# A collection is like a table in a regular database — it groups related vectors.
# get_or_create means: if "legal_docs" already exists (from a previous run),
# use it. If not, create it fresh. This makes the script safely re-runnable.
chroma_collection = chroma_client.get_or_create_collection("legal_docs")

# Wrap the ChromaDB collection in LlamaIndex's adapter class so LlamaIndex
# knows how to read from and write to it.
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# StorageContext bundles together all storage-related configuration.
# Here we're telling LlamaIndex: "when you need to store vectors, use
# this ChromaDB vector store" — rather than the default in-memory storage.
storage_context = StorageContext.from_defaults(vector_store=vector_store)


# --- STEP 3: EMBED AND INDEX ---
print("⚙️  Embedding and indexing (this may take a few minutes)...")

# This is where the real work happens. For each document loaded in Step 1,
# LlamaIndex will:
#   1. Split it into smaller chunks (e.g. 512 tokens each with some overlap)
#   2. Send each chunk to nomic-embed-text to get a vector (a list of ~768 numbers)
#   3. Store that vector + the original text + metadata into ChromaDB on disk
#
# show_progress=True displays a progress bar so you can see it working.
# This step can take 1–5 minutes depending on how many documents you have.
# You only need to re-run this when your documents change.
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
    transformations=[SentenceSplitter(
        chunk_size=512,        # max tokens per chunk
        chunk_overlap=50       # overlap between chunks for context continuity
    )],
    show_progress=True
)

# Once this line prints, everything is saved to ./chroma_db on disk.
# The app.py can now load that index and start answering questions.
print("✅ Ingestion complete! Your documents are indexed in ./chroma_db")