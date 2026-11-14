"""Semantic search over the ChromaDB collections.

Understanding the scores
------------------------
ChromaDB is configured with cosine space and returns a **distance**::

    distance   = 1 - cosine_similarity          (smaller = closer, range 0..2)

This module converts it to a **cosine similarity**::

    similarity = 1 - distance                    (larger = closer, range -1..1)

The similarity is a cosine score, *not* a percentage and *not* a probability. Only the
ranking and relative differences are meaningful; see the README for guidance.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
from src.embeddings import Embedder, get_embedder
from src.utils import distance_to_similarity, get_logger
from src.vector_store import CollectionNotFoundError, VectorStore

logger = get_logger(__name__)

SearchHit = Dict[str, Any]


class InvalidQueryError(ValueError):
    """The query or one of the search arguments is invalid."""


_default_store: Optional[VectorStore] = None
_default_store_path: Optional[Path] = None


def get_default_store() -> VectorStore:
    """Lazily open the project's persistent database (re-opened if the path changed)."""
    global _default_store, _default_store_path
    if _default_store is None or _default_store_path != config.CHROMA_DIR:
        _default_store = VectorStore(config.CHROMA_DIR)
        _default_store_path = config.CHROMA_DIR
    return _default_store


def reset_default_store() -> None:
    """Drop the cached default store (used by tests and after resetting the database)."""
    global _default_store, _default_store_path
    if _default_store is not None:
        _default_store.close()
    _default_store = None
    _default_store_path = None


def validate_search_args(query: Any, top_k: Any, strategy: Any) -> Tuple[str, int, str]:
    """Validate and normalise arguments; raises InvalidQueryError with a helpful message."""
    if not isinstance(query, str):
        raise InvalidQueryError(f"query must be a string, got {type(query).__name__}")
    cleaned = " ".join(query.split())
    if not cleaned:
        raise InvalidQueryError("query must not be empty")
    if len(cleaned) > config.MAX_QUERY_CHARS:
        raise InvalidQueryError(f"query is too long ({len(cleaned)} characters, maximum {config.MAX_QUERY_CHARS})")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise InvalidQueryError(f"top_k must be an integer, got {top_k!r}")
    if not 1 <= top_k <= config.MAX_TOP_K:
        raise InvalidQueryError(f"top_k must be between 1 and {config.MAX_TOP_K}, got {top_k}")
    if strategy not in config.STRATEGIES:
        raise InvalidQueryError(f"strategy must be one of {list(config.STRATEGIES)}, got {strategy!r}")
    return cleaned, top_k, strategy


def search(
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    strategy: str = config.STRATEGY_RECURSIVE,
    store: Optional[VectorStore] = None,
    embedder: Optional[Embedder] = None,
) -> List[SearchHit]:
    """Return the ``top_k`` chunks most similar to ``query``, best first.

    Args:
        query: natural-language question.
        top_k: number of results (1..config.MAX_TOP_K). Fewer are returned if the index is smaller.
        strategy: ``"fixed"`` or ``"recursive"`` - which chunking strategy's index to search.
        store / embedder: optional overrides (mainly for tests); defaults use the project database.

    Returns:
        A list of dicts sorted by ``rank`` (1 = best) with keys: rank, similarity (cosine,
        -1..1, higher is better), distance (raw Chroma cosine distance, lower is better),
        document_name, document_id, page_number, chunk_id, chunk_index, chunking_strategy,
        source_file, text.

    Raises:
        InvalidQueryError: bad query / top_k / strategy.
        CollectionNotFoundError: the strategy has not been indexed (or is empty).
        EmbeddingMismatchError: the index was built with a different embedding model.
    """
    cleaned, top_k, strategy = validate_search_args(query, top_k, strategy)
    vector_store = store or get_default_store()
    model = embedder or vector_store.embedder or get_embedder()

    query_vector = model.embed([cleaned])[0]
    raw = vector_store.query(strategy, query_vector, top_k)

    ids = raw["ids"][0]
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    hits: List[SearchHit] = []
    for position, (chunk_id, text, meta, distance) in enumerate(zip(ids, documents, metadatas, distances), start=1):
        hits.append(
            {
                "rank": position,
                "similarity": round(distance_to_similarity(distance), 4),
                "distance": round(float(distance), 4),
                "document_name": meta.get("document_name", ""),
                "document_id": meta.get("document_id", ""),
                "page_number": int(meta.get("page_number", 0)),
                "chunk_id": chunk_id,
                "chunk_index": int(meta.get("chunk_index", 0)),
                "chunking_strategy": meta.get("chunking_strategy", strategy),
                "source_file": meta.get("source_file", ""),
                "text": text,
            }
        )
    # Chroma already sorts by distance; sort again defensively so rank is always consistent.
    hits.sort(key=lambda h: h["distance"])
    for position, hit in enumerate(hits, start=1):
        hit["rank"] = position
    return hits


def timed_search(
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    strategy: str = config.STRATEGY_RECURSIVE,
    store: Optional[VectorStore] = None,
    embedder: Optional[Embedder] = None,
) -> Tuple[List[SearchHit], float]:
    """Like :func:`search` but also returns the wall-clock time in milliseconds
    (query embedding + vector lookup)."""
    started = time.perf_counter()
    hits = search(query, top_k, strategy, store=store, embedder=embedder)
    return hits, (time.perf_counter() - started) * 1000.0


__all__ = [
    "search", "timed_search", "InvalidQueryError", "CollectionNotFoundError",
    "get_default_store", "reset_default_store", "validate_search_args",
]
