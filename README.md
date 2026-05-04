# Legal AI RAG System — Local PoC

A fully local Retrieval-Augmented Generation (RAG) system that lets you 
query a knowledge base of legal documents, firm SOPs, and case files using 
a locally hosted LLM. No data leaves your machine.

Built with [Ollama](https://ollama.com), [LlamaIndex](https://www.llamaindex.ai), 
[ChromaDB](https://www.trychroma.com), and [Streamlit](https://streamlit.io).

---

## How it works

1. Documents are parsed, chunked, and embedded into a local vector database (ChromaDB)
2. When you ask a question, the app finds the most semantically relevant chunks
3. Those chunks are passed to a local LLM (llama3.2:3b) which generates a cited answer
4. Nothing is sent to any external API or service

---

## Requirements

- macOS with Apple Silicon (tested on M3, 8GB unified memory)
- Python 3.11+
- [Homebrew](https://brew.sh)

---

## Setup

### 1. Install and start Ollama

```bash
brew install ollama
ollama serve
```

### 2. Pull the required models

In a new terminal tab:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Verify the LLM works:

```bash
ollama run llama3.2:3b "What is a limitation period?"
```

Type `/bye` to exit.

### 3. Set up the Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

When working in different branches, you must install the requirements again.

If you already have the python venv set up with the requirements, only run:

```bash
source venv/bin/activate
```

To exit the venv, run:

```bash
deactivate
```

---

## Running the app

### 1. Add your documents

Create the following folder structure and add your `.txt`, `.pdf`, or `.docx` files:

```
documents/
├── laws/       ← Legislation and regulatory summaries
├── sops/       ← Firm procedures and standards
└── clients/    ← Case files
```

### 2. Index your documents

```bash
python ingest.py
```

Re-run this any time you add or update documents.

### 3. Launch the app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## Project structure

```
local-legal-RAG-system/
├── documents/        ← Your source documents (not committed)
├── ingest.py         ← Parses and indexes documents into ChromaDB
├── app.py            ← Streamlit chat interface
├── requirements.txt  ← Python dependencies
├── .gitignore
└── README.md
```

> `chroma_db/` and `venv/` are excluded from version control via `.gitignore`.
> Run `ingest.py` after cloning to regenerate the vector database locally.

---

## Notes

- Model: `llama3.2:3b` — approximately 2GB download, runs on 8GB unified memory
- This is a proof-of-concept. Do not use with real regulated data (PHI, PII, 
  client files) without a production-grade, air-gapped deployment
- All AI responses should be reviewed by a licensed lawyer before being acted upon