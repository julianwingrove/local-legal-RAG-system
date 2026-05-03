# ingest.py — Run this once to index your documents

import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

print("🔍 Loading documents...")
documents = SimpleDirectoryReader(
    "./documents",
    recursive=True,
    required_exts=[".txt", ".pdf", ".docx"]
).load_data()
print(f"✅ Loaded {len(documents)} document chunks")

print("🔢 Setting up vector store...")
embed_model = OllamaEmbedding(model_name="nomic-embed-text")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("legal_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("⚙️  Embedding and indexing (this may take a few minutes)...")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True
)

print("✅ Ingestion complete! Your documents are indexed in ./chroma_db")