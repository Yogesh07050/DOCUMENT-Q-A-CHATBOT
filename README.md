# Doc-QnA-Chatbot (Advanced RAG)

Production-ready Retrieval-Augmented Generation for document Q&A with FastAPI, FAISS, sentence-transformers embeddings, and a Streamlit chat UI.

## Layout
- `backend/`: FastAPI API (`/ingest`, `/query`, `/health`), embeddings, vector store persistence.
- `frontend/`: Streamlit interface that uploads docs and shows answers with supporting context.
- `data/`: Persistent storage for uploads and FAISS artifacts.

## Setup
1) Create a virtual environment at the repo root and install all dependencies from the single `requirements.txt`:
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Configure `.env` at repository root (copy from `.env` template):
```
OPENAI_API_KEY=your_key_here   # optional, enables LLM generation
API_BASE=http://localhost:8000
```

3) Start the backend (production-ready flags):
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 2
```

4) Run the Streamlit frontend (uses the same environment):
```bash
streamlit run frontend/app.py
```

## How it works
- Upload PDF/TXT/MD files; documents are chunked with recursive text splitter.
- Chunks are embedded via `sentence-transformers/all-MiniLM-L6-v2`, normalized, and stored in a FAISS IndexIDMap.
- `/query` embeds the question, retrieves top-k chunks, and (optionally) sends context to OpenAI for grounded answers.
- Responses include the answer plus scored contexts for traceability.

## Operations
- Reset storage (clears FAISS index and metadata):
```bash
python -c "from backend.storage import reset_store; reset_store()"
```
- Health check:
```bash
curl http://localhost:8000/health
```

## Notes
- Set `OPENAI_API_KEY` to enable generation; otherwise the API returns context-only guidance.
- Tune `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `TOP_K` in `.env` to adjust retrieval behavior.

## OUTPUT:

<img width="1918" height="921" alt="image" src="https://github.com/user-attachments/assets/cbeadfb9-558c-4f8b-b9d2-4e6754699411" />

