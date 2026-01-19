"""
Utilities for persisting FAISS indexes and chunk metadata.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np

from .config import settings, ensure_directories


def _empty_index(dim: int) -> faiss.IndexIDMap:
    """Create a new index ready for cosine similarity search."""
    inner = faiss.IndexFlatIP(dim)
    return faiss.IndexIDMap(inner)


def load_index(dim: int) -> faiss.IndexIDMap:
    """
    Load an existing FAISS index or create a new one if missing.
    Raises a ValueError when dimension mismatches to prevent silent corruption.
    """
    ensure_directories()
    path = settings.index_path
    if not path.exists():
        return _empty_index(dim)

    index = faiss.read_index(str(path))
    if index.d != dim:
        raise ValueError(
            f"Existing index dimension ({index.d}) does not match embedding dimension ({dim}). "
            "Delete the index or align embedding configuration."
        )
    if not isinstance(index, faiss.IndexIDMap):
        index = faiss.IndexIDMap(index)
    return index


def persist_index(index: faiss.IndexIDMap) -> None:
    ensure_directories()
    faiss.write_index(index, str(settings.index_path))


def load_metadata() -> List[Dict[str, Any]]:
    ensure_directories()
    meta_path = settings.metadata_path
    if not meta_path.exists():
        return []
    with open(meta_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def persist_metadata(metadata: List[Dict[str, Any]]) -> None:
    ensure_directories()
    with open(settings.metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)


def reset_store() -> None:
    """Remove persisted index and metadata. Useful for rebuilds."""
    for path in (settings.index_path, settings.metadata_path):
        Path(path).unlink(missing_ok=True)
