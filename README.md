# local-legal-RAG-system
This is a locally hosted RAG system PoC to demonstrate how one might work when combining company SOPs and local regulations.

This specific RAG runs on ollama llama3.2:3b. This model will run smoothly on an 8GB unified memory system. The download is roughly 2.5GB total.

To set up this program:

1. brew install ollama

2. ollama serve

3. go to new terminal tab and run: ollama pull llama3.2:3b

4. ollama pull nomic-embed-text

5. test that it works: ollama run llama3.2:3b "[ask a question]"

To run this program:

1. create your document repo at /documents/

2.run the ingestion: python ingest.py

3. launch the app: streamlit run app.py