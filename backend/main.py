"""
Entry point for the backend API server.
Provides ingestion and query endpoints for the RAG pipeline.
"""
from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware

from backend_core.config import ensure_directories, settings
from .query import query_pipeline
from .ingest import ingest_documents

app = FastAPI(title="Doc QnA Chatbot", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    ensure_directories()
    return {"status": "ok", "top_k": settings.top_k}


@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    return ingest_documents(file)


@app.post("/query")
def query_endpoint(question: str = Query(..., description="Natural language question to ask.")):
    return query_pipeline(question)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
