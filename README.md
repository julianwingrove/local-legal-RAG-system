# Legal AI RAG System — Local PoC

A fully local Retrieval-Augmented Generation (RAG) system that lets you
query a focused knowledge base of legal documents, firm SOPs, and case
files using a locally hosted LLM. No data leaves your machine.

Built with [Ollama](https://ollama.com), [LlamaIndex](https://www.llamaindex.ai),
[ChromaDB](https://www.trychroma.com), and [Streamlit](https://streamlit.io).

---

## How it works

1. Documents are parsed, split into 150-token chunks, embedded, and stored
   in a local vector database (ChromaDB)
2. When you ask a question, a multi-category retriever searches legislation,
   SOPs, and client files independently and combines the results
3. Up to 30 retrieved chunks are passed to a local LLM within a 6,500 token
   context window
4. The LLM generates a cited answer grounded in your documents
5. Nothing is sent to any external API or service

---

## Architecture

```
Your question
↓
Multi-category retriever
↓
┌──────────────┬──────────────┬───────────────┐
│  Laws (×12)  │  SOPs (×10)  │ Clients (×8)  │
└──────────────┴──────────────┴───────────────┘
↓
Up to 30 chunks × 150 tokens = 4,500 tokens of content
↓
llama3.2:3b (6500 token context window)
↓
Cited answer
```

Each category is searched independently so large legislation PDFs cannot
crowd out smaller SOP and client file content.

---

## Current document corpus

```
documents/
├── laws/
│   └── limitations_act_2002.txt
├── sops/
│   └── sop_002_limitation_period_management.txt
└── clients/
└── case_margaret_chen.txt
```
These three documents are tightly interconnected — the Limitations Act
drives the urgency of the client file, and SOP-002 governs the required
response procedure.

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

The documents folder is not committed to the repo. Create it locally:

```bash
mkdir -p documents/laws documents/sops documents/clients
```

Add your files to the correct subfolders. The ingestion pipeline
automatically tags documents by category based on subfolder name.

> PDFs must be digital (text-selectable), not scanned images.
> Test by trying to highlight and copy text in Preview.
> If you cannot select text, the PDF cannot be parsed.

### 5. Index your documents

Ollama must be running before ingestion. Then:

```bash
python ingest.py
```

Re-run any time you add or update documents. Always delete
`chroma_db/` first to force a clean re-index:

```bash
rm -rf chroma_db/
python ingest.py
```

> If you change the embedding model, you must delete chroma_db/ and
> re-ingest. Vectors from different models are incompatible.

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

The app opens automatically at `http://localhost:8501`.

> VS Code auto-activates the venv when you open a new terminal
> in the project folder — no need to run activate manually.

---

## Login

| Username | Password |
|----------|----------|
| admin    | legal123 |

To add or change credentials, update the `CREDENTIALS` dictionary in `app.py`.

---

## Features

**Multi-conversation sidebar**
Create, switch between, and delete independent chat sessions.
Each conversation is auto-named from the first message.

**Multi-category query router**
Legislation, SOPs, and client files are searched independently on every
query — no category can crowd out another regardless of document size.

**Source citations**
Every response shows which documents were retrieved, making it easy to
verify answers against source material.

**No chat history**
Each question is treated as self-contained. The full 6,500 token context
window is reserved for document content, maximising what the LLM can see
per query and avoiding stale context from previous exchanges.

**Login screen**
Username and password required on every session.

---

## Optimised retrieval settings

| Setting | Value | Reason |
|---|---|---|
| Chunk size | 100 tokens | Isolates individual sections into dedicated chunks |
| Chunk overlap | 15 tokens | Preserves context at chunk boundaries |
| Context window | 6,500 tokens | Safe ceiling for 8GB unified memory |
| Law top_k | 20 | Large PDFs need more retrieval slots |
| SOP top_k | 18 | Covers nearly all of a medium SOP document |
| Client top_k | 14 | Covers virtually the entire client file |
| Embedding model | nomic-embed-text | 274MB, 768 dimensions, 2048 token limit |
| LLM | llama3.2:3b | ~2GB, fits comfortably within 8GB |

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

> `chroma_db/` and `venv/` are gitignored — regenerate locally after cloning.
> `documents/` is gitignored — add your own files after cloning.

---

## Git workflow

```
main    ← stable, always working
dev     ← active development
```

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
|---|---|
| `source venv/bin/activate` | Start of every working session |
| `deactivate` | When you are done working |
| `pip install -r requirements.txt` | First-time setup, or after pulling changes that update requirements.txt |
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
| `ModuleNotFoundError` | Venv not active | Run `source venv/bin/activate` |
| `input length exceeds context length` | Chunk too large for embedding model | Reduce `chunk_size` in `ingest.py` |
| Garbled text in responses | PDF encoding issue | Already handled by pymupdf |
| Empty or wrong answers | Document not indexed | Run `rm -rf chroma_db/ && python ingest.py` |

---

## Known limitations (PoC)

- `llama3.2:3b` occasionally confuses section numbers between documents
  when multiple documents are in context simultaneously
- The model sometimes provides correct answers with fabricated reasoning
- Response times are 10–30 seconds depending on query complexity
- Context window capped at 6500 tokens to avoid OOM on 8GB
- No chat history — each question must be self-contained

These are expected at PoC scale and are addressed in the production build.

---

## Production roadmap

The production system replaces the PoC components as follows:

| Component | PoC | Production |
|---|---|---|
| Hardware | MacBook Pro M3 8GB | DGX Spark |
| LLM | llama3.2:3b | Nemotron 3 Super 120B |
| Embedding Model | nomic-embed-text | NV-Embed-v2 |
| Context window | 6500 tokens | 32,768+ tokens |
| Document corpus | 3 documents | Hundreds of files + CanLII API |
| Case law | Not included | Live via CanLII API |
| Authentication | Hardcoded credentials | SSO with MFA |
| Deployment | Local Streamlit | Air-gapped on-premise server |

---

## Notes

- Ollama must be running before launching the app or running ingestion
- This is a proof-of-concept. Do not use with real regulated data (PHI,
  PII, client files) without a production-grade, air-gapped deployment
- All AI responses must be reviewed by a licensed lawyer before being
  acted upon