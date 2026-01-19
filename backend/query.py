"""
Query pipeline for RAG.
"""
from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
from fastapi import HTTPException
from openai import OpenAI

from backend_core.config import settings
from backend_core.embedding import embed_texts
from backend_core.storage import load_index, load_metadata


def _generate_answer(question: str, contexts: List[str]) -> str:
    """Generate an answer using OpenAI if configured, otherwise return a grounded summary."""
    if not contexts:
        return "No relevant context found for this question."

    context_block = "\n\n".join(f"- {ctx}" for ctx in contexts)
    if settings.openai_api_key:
        client = OpenAI(api_key=settings.openai_api_key)
        completion = client.chat.completions.create(
            model=settings.generator_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise assistant that answers using only the provided context.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_block}\n\nQuestion: {question}",
                },
            ],
            temperature=0.1,
        )
        return completion.choices[0].message.content.strip()

    return (
        "Generation model not configured. Relevant context:\n"
        f"{context_block}\n\nSet OPENAI_API_KEY in .env to enable LLM answers."
    )


def query_pipeline(question: str) -> Dict[str, Any]:
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    # Load store
    metadata = load_metadata()
    if not metadata:
        raise HTTPException(status_code=400, detail="No documents have been ingested yet.")

    # Encode question
    q_emb = embed_texts([question])
    if q_emb.size == 0:
        raise HTTPException(status_code=500, detail="Failed to embed the question.")

    # Retrieve
    index = load_index(q_emb.shape[1])
    if index.ntotal == 0:
        raise HTTPException(status_code=400, detail="The vector index is empty. Ingest documents first.")

    scores, ids = index.search(q_emb, settings.top_k)
    hit_ids = ids[0]
    hit_scores = scores[0]

    results = []
    contexts = []
    for idx, score in zip(hit_ids, hit_scores):
        if idx == -1:
            continue
        if idx >= len(metadata):
            continue
        meta = metadata[idx]
        meta["score"] = float(score)
        results.append(meta)
        contexts.append(meta["text"])

    answer = _generate_answer(question, contexts)
    return {"answer": answer, "contexts": results}
