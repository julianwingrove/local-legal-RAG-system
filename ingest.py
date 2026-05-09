# ingest.py — Document ingestion pipeline with category tagging
# Run this once after adding or updating documents.
# Re-running will re-index everything from scratch.

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

        # Determine category based on which subfolder the file lives in.
        # This is used later for per-category retrieval in the query router.
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
                # Use pymupdf for robust PDF text extraction.
                # Handles complex encoding that pypdf cannot.
                doc = fitz.open(filepath)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()

                if text.strip():
                    documents.append(Document(
                        text=text,
                        metadata={
                            "file_name": filename,
                            "file_path": filepath,
                            "category": category
                        }
                    ))
                else:
                    print(f"⚠️  No text extracted from {filename} — skipping")

            elif ext in [".txt", ".docx"]:
                # Use SimpleDirectoryReader for plain text and Word docs.
                from llama_index.core import SimpleDirectoryReader
                docs = SimpleDirectoryReader(
                    input_files=[filepath]
                ).load_data()
                for d in docs:
                    d.metadata["file_name"] = filename
                    d.metadata["file_path"] = filepath
                    d.metadata["category"] = category
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

# nomic-embed-text runs locally via Ollama — no internet required.
# Must use the same model here and in app.py — mixing models breaks search.
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Connect to ChromaDB on disk. PersistentClient survives after the script ends.
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# get_or_create_collection: connect if exists, create fresh if not.
chroma_collection = chroma_client.get_or_create_collection("legal_docs")

# Bridge between LlamaIndex and ChromaDB.
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Tell LlamaIndex to store vectors in ChromaDB rather than in-memory.
storage_context = StorageContext.from_defaults(vector_store=vector_store)


# --- STEP 3: CHUNK, EMBED, AND INDEX ---
print("⚙️  Embedding and indexing (this may take several minutes)...")

# SentenceSplitter breaks each document into overlapping chunks before embedding.
# chunk_size=512: max tokens per chunk — stays within nomic-embed-text's limit.
# chunk_overlap=50: consecutive chunks share 50 tokens so answers aren't
# cut off at chunk boundaries.
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
    transformations=[SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )],
    show_progress=True
)

print("✅ Ingestion complete! Your documents are indexed in ./chroma_db")