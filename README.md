# local-legal-RAG-system
This is a locally hosted RAG system PoC to demonstrate how one might work when combining company SOPs and local regulations.

This specific RAG runs on ollama llama3.2:3b. This model will run smoothly on an 8GB unified memory system. The download is roughly 2.5GB total.

To set up this program:

1. brew install ollama

2. ollama serve

3. go to new terminal tab and run: ollama pull ollama pull llama3.2:3b

4. ollama pull nomic-embed-text

5. test that it works: ollama run llama3.2:3b "[ask a question]"

If you want to run in an isolated environment run:

1. python3 -m venv venv
source venv/bin/activate

2. pip install \
  llama-index-core \
  llama-index-llms-ollama \
  llama-index-embeddings-ollama \
  llama-index-vector-stores-chroma \
  chromadb \
  streamlit \
  pypdf \
  docx2txt

To run this program:

1. create your document repo at /documents/

2. run the ingestion: python ingest.py

3. launch the app: streamlit run app.py

The folder structure should look like this:

legal-ai-poc/
├── documents/
│   ├── laws/          ← Ontario statutes, summaries
│   ├── sops/          ← Firm procedures
│   └── clients/       ← Case files
├── chroma_db/         ← Auto-created after ingest.py
├── venv/              ← Python virtual environment
├── ingest.py          ← Run once per document update
├── app.py             ← The chat UI
└── requirements.txt   ← pip freeze > requirements.txt