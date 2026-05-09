# Legal AI RAG System — Local PoC

A fully local Retrieval-Augmented Generation (RAG) system that lets you
query a knowledge base of legal documents, firm SOPs, and case files using
a locally hosted LLM. No data leaves your machine.

Built with [Ollama](https://ollama.com), [LlamaIndex](https://www.llamaindex.ai),
[ChromaDB](https://www.trychroma.com), and [Streamlit](https://streamlit.io).

---

## How it works

1. Documents are parsed, chunked, and embedded into a local vector database (ChromaDB)
2. When you ask a question, a multi-category query router searches laws, SOPs,
   and client files independently, then combines the results
3. Those combined results are passed to a local LLM (llama3.2:3b) along with
   recent conversation history for context-aware answers
4. The LLM generates a cited answer grounded in your documents
5. Nothing is sent to any external API or service

---

## Architecture

```
Your question + recent chat history
↓
Multi-category query router
↓
┌─────────────┬─────────────┬──────────────┐
│  Laws (x10) │  SOPs (x5)  │ Clients (x5) │
└─────────────┴─────────────┴──────────────┘
↓
Combined results (up to 20 chunks)
↓
LLM generates cited answer
```

Each category is searched independently so no category can crowd out another.
A question spanning multiple categories (e.g. legislation + SOP + client file)
gets fair representation from all three.

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

### 3. Clone the repo and set up Python environment

```bash
git clone https://github.com/julianwingrove/local-legal-RAG-system.git
cd local-legal-RAG-system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Add your documents

Add your `.txt`, `.pdf`, or `.docx` files to the documents folder.
The folder is not committed to the repo — create it locally:

```bash
mkdir -p documents/laws documents/sops documents/clients
```

Subfolders under `laws/` are supported and recommended:

```
documents/
├── laws/
│   ├── federal/        ← Federal legislation
│   └── ontario/        ← Ontario statutes and regulations
├── sops/               ← Firm procedures and standards
└── clients/            ← Case files
```

> PDFs must be digital (text-selectable), not scanned images.
> To test, try highlighting and copying text in Preview.
> If you cannot select text, the PDF cannot be parsed.

The ingestion pipeline automatically tags documents by category based on
which subfolder they live in. This powers the multi-category query router.

### 5. Index your documents

Ollama must be running before ingestion. Then:

```bash
python ingest.py
```

This may take several minutes depending on the size of your document library.
Re-run any time you add or update documents — delete `chroma_db/` first
to force a full re-index:

```bash
rm -rf chroma_db/
python ingest.py
```

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

| Username | Password  |
|----------|-----------|
| admin    | legal123  |

To add or change credentials, update the `CREDENTIALS` dictionary in `app.py`.

---

## Features

**Multi-conversation sidebar**
- Create multiple independent chat sessions
- Each conversation maintains its own context and history
- Conversations are auto-named from the first message
- Delete individual conversations or clear all

**Chat memory**
- The last 3 exchanges are injected into each prompt
- Follow-up questions work without repeating context
- e.g. "What is the limitation period for his tort claim?" after
  asking about a specific client

**Multi-category query router**
- Laws, SOPs, and client files are searched independently
- Results are combined before being sent to the LLM
- Cross-category questions work reliably

**Source citations**
- Every response shows which documents were retrieved
- Helps verify the answer is grounded in your documents

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
> `documents/` is also excluded — add your own files locally after cloning.
> Run `ingest.py` after cloning to build the vector database locally.

---

## Git workflow
main    ← stable, always working
dev     ← active development

```bash
# Daily work on dev
git checkout dev
git add .
git commit -m "describe what changed"
git push

# Merge to main when stable
git checkout main
git merge dev
git push
git checkout dev
```

---

## Venv reference

| Command | When to use |
|---------|-------------|
| `source venv/bin/activate` | Start of every session |
| `deactivate` | When you are done working |
| `pip install -r requirements.txt` | First time setup, or after pulling changes that update requirements.txt |
| `pip freeze > requirements.txt` | After installing any new package |

---

## Debugging

**Check what is indexed:**
```bash
python3 -c "
import chromadb
client = chromadb.PersistentClient(path='./chroma_db')
col = client.get_or_create_collection('legal_docs')
results = col.get(include=['metadatas'])
files = sorted(set(m.get('file_name', 'unknown') for m in results['metadatas']))
print(f'Total chunks: {len(results[\"metadatas\"])}')
for f in files:
    print(f'  - {f}')
"
```

**Common issues:**

| Error | Cause | Fix |
|---|---|---|
| `ConnectionError: Failed to connect to Ollama` | Ollama not running | Run `ollama serve` in a separate terminal |
| `Empty response` | Document not indexed | Re-run `python ingest.py` |
| `ModuleNotFoundError` | Venv not active | Run `source venv/bin/activate` |
| `the input length exceeds context length` | Chunk too large | Already handled by SentenceSplitter in ingest.py |
| Garbled text in responses | PDF encoding issue | Already handled by pymupdf in ingest.py |

---

## Known limitations (PoC)

- `llama3.2:3b` struggles to independently identify applicable legislation
  from plain-language case files — a model size limitation
- The model occasionally hallucinates citations or section numbers
- Response times are 10–30 seconds depending on query complexity
- Context window is capped at 3,072 tokens to avoid OOM crashes on 8GB

These are expected at PoC scale. The production deployment uses a larger
model on dedicated hardware which addresses all of the above.

---

## Notes

- Model: `llama3.2:3b` — approximately 2GB download, runs on 8GB unified memory
- Ollama must be running before launching the app or running ingestion
- This is a proof-of-concept. Do not use with real regulated data (PHI, PII,
  client files) without a production-grade, air-gapped deployment
- All AI responses should be reviewed by a licensed lawyer before being acted upon