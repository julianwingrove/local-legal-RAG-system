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

## First-time setup

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

### 4. Add your documents

Add your `.txt`, `.pdf`, or `.docx` files to the documents folder.
Subfolders are supported and recommended for organisation:

```
documents/
├── laws/
│   ├── federal/        ← Federal legislation
│   └── ontario/        ← Ontario statutes and regulations
├── sops/               ← Firm procedures and standards
└── clients/            ← Case files
```

> PDFs must be digital (text-selectable) not scanned images.
> To test, try highlighting and copying text in Preview.
> If you cannot select text, the PDF cannot be parsed.

### 5. Index your documents

Ollama must be running before ingestion. Then:

```bash
python ingest.py
```

This may take several minutes depending on the size of your document library.
Re-run any time you add or update documents.

---

## Daily startup

**Terminal tab 1 — start Ollama:**
```bash
ollama serve
```

**Terminal tab 2 — start the app:**
```bash
cd local-legal-RAG-system
source venv/bin/activate
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

> If you are using VS Code, the virtual environment activates automatically
> when you open a new terminal in the project folder.

---

## Login

The app requires a username and password. Default credentials:

| Username | Password |
|----------|----------|
| admin | legal123 |

To add or change credentials, update the `CREDENTIALS` dictionary in `app.py`.

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

## Venv reference

| Command | When to use |
|---------|-------------|
| `source venv/bin/activate` | Start of every session |
| `deactivate` | When you are done working |
| `pip install -r requirements.txt` | First time setup only, or after pulling changes that update requirements.txt |
| `pip freeze > requirements.txt` | After installing any new package |

---

## Notes

- Model: `llama3.2:3b` — approximately 2GB download, runs on 8GB unified memory
- Ollama must be running before launching the app or running ingestion
- This is a proof-of-concept. Do not use with real regulated data (PHI, PII,
  client files) without a production-grade, air-gapped deployment
- All AI responses should be reviewed by a licensed lawyer before being acted upon