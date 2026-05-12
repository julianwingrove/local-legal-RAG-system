# ingest.py — Document ingestion pipeline
# Run this once after adding or updating documents.
# Delete chroma_db/ and re-run if you change the embedding model or chunk size.

import os
import fitz  # pymupdf
import chromadb

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


# --- STEP 1: LOAD AND PARSE DOCUMENTS ---
print("🔍 Loading documents...")
documents = []

for root, dirs, files in os.walk("./documents"):
    for filename in files:
        filepath = os.path.join(root, filename)
        ext = os.path.splitext(filename)[1].lower()

        # Tag each document by category based on subfolder.
        # Used by the multi-category retriever in app.py to ensure
        # each category gets fair representation on every query.
        if "clients" in filepath:
            category = "client"
        elif "sops" in filepath:
            category = "sop"
        elif "laws" in filepath:
            category = "law"
        else:
            category = "general"

        try:
            if ext == ".pdf":
                # pymupdf extracts clean text from digital PDFs.
                # Handles complex encoding that pypdf cannot.
                doc = fitz.open(filepath)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()

                if text.strip():
                    # Only file_name and category stored in metadata.
                    # file_path excluded to prevent it leaking into responses.
                    documents.append(Document(
                        text=text,
                        metadata={
                            "file_name": filename,
                            "category": category
                        }
                    ))
                else:
                    print(f"⚠️  No text extracted from {filename} — skipping")

            elif ext in [".txt", ".docx"]:
                from llama_index.core import SimpleDirectoryReader
                docs = SimpleDirectoryReader(
                    input_files=[filepath]
                ).load_data()
                for d in docs:
                    # Only store file_name and category — no file_path.
                    d.metadata = {
                        "file_name": filename,
                        "category": category
                    }
                documents.extend(docs)

        except Exception as e:
            print(f"⚠️  Failed to load {filename}: {e}")

print(f"✅ Loaded {len(documents)} documents")
print(f"   Categories: "
      f"{sum(1 for d in documents if d.metadata.get('category') == 'law')} laws, "
      f"{sum(1 for d in documents if d.metadata.get('category') == 'sop')} SOPs, "
      f"{sum(1 for d in documents if d.metadata.get('category') == 'client')} client files")


# --- STEP 2: SET UP VECTOR STORE ---
print("🔢 Setting up vector store...")

# nomic-embed-text: 2048 token context limit, 768-dimension vectors.
# Must match the embedding model used in app.py — mixing models
# produces garbage retrieval results.
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("legal_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


# --- STEP 3: CHUNK, EMBED, AND INDEX ---
print("⚙️  Embedding and indexing (this may take a few minutes)...")

# chunk_size=100: small enough to isolate individual SOP sections and
# legislative provisions into their own chunks, enabling precise retrieval.
# At 100 tokens, each numbered section (e.g. s.3.3, s.4.2) typically
# gets its own dedicated chunk rather than sharing with adjacent sections.
#
# chunk_overlap=15: small overlap preserves context at chunk boundaries
# without wasting significant token budget.
#
# nomic-embed-text has a 2048 token limit — 100 tokens is well within it.
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
    transformations=[SentenceSplitter(
        chunk_size=100,
        chunk_overlap=15
    )],
    show_progress=True
)

print("✅ Ingestion complete! Your documents are indexed in ./chroma_db")