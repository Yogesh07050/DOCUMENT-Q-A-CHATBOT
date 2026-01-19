"""
Document ingestion and chunking logic.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Any

import numpy as np
from fastapi import HTTPException, UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from backend_core.config import settings, ensure_directories
from backend_core.embedding import embed_texts
from backend_core.storage import load_index, persist_index, load_metadata, persist_metadata


def _read_text_from_file(raw_bytes: bytes, extension: str) -> str:
    """Parse supported file types into text."""
    if extension == ".pdf":
        pdf = PdfReader(io.BytesIO(raw_bytes))
        return "\n".join(page.extract_text() or "" for page in pdf.pages)
    return raw_bytes.decode("utf-8", errors="ignore")


def ingest_documents(file: UploadFile) -> Dict[str, Any]:
    ensure_directories()
    extension = Path(file.filename).suffix.lower()
    if extension not in settings.allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {extension}")

    raw_bytes = file.file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise HTTPException(status_code=400, detail=f"File exceeds {settings.max_file_size_mb} MB limit.")

    upload_path = settings.upload_dir / file.filename
    upload_path.write_bytes(raw_bytes)

    text = _read_text_from_file(raw_bytes, extension).strip()
    if not text:
        raise HTTPException(status_code=400, detail="No readable text found in the uploaded file.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunk_texts = splitter.split_text(text)
    if not chunk_texts:
        raise HTTPException(status_code=400, detail="Failed to split document into chunks.")

    embeddings = embed_texts(chunk_texts)
    if embeddings.size == 0:
        raise HTTPException(status_code=500, detail="Embedding model returned empty vectors.")

    index = load_index(embeddings.shape[1])
    metadata = load_metadata()

    start_id = len(metadata)
    ids = np.arange(start_id, start_id + len(chunk_texts))
    index.add_with_ids(embeddings, ids)

    for idx, text_chunk in zip(ids, chunk_texts):
        metadata.append(
            {
                "id": int(idx),
                "source": file.filename,
                "chunk": int(idx) - start_id + 1,
                "text": text_chunk,
            }
        )

    persist_index(index)
    persist_metadata(metadata)

    return {
        "file": file.filename,
        "chunks_indexed": len(chunk_texts),
        "total_chunks": len(metadata),
        "index_path": str(settings.index_path),
    }
