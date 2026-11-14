"""Persistent ChromaDB storage for chunks, embeddings and metadata.

Layout: one ChromaDB *collection per chunking strategy* (``docsearch_fixed`` and
``docsearch_recursive``) inside a single on-disk database, so the two strategies are
always separately identifiable and can be queried independently.

Every collection uses cosine space. ChromaDB therefore returns a cosine *distance*
(``1 - cosine similarity``); conversion to a similarity score happens in ``search.py``.

Re-ingesting is safe: chunk IDs are deterministic and a document's previous chunks are
removed before its new chunks are upserted, so no duplicates are ever created.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import chromadb
from chromadb.errors import NotFoundError

import config
from src.chunking import Chunk, chunk_document
from src.document_loader import Document
from src.embeddings import Embedder, get_embedder
from src.utils import get_logger

logger = get_logger(__name__)

UPSERT_BATCH = 256


class CollectionNotFoundError(LookupError):
    """The requested strategy has not been indexed yet."""


class EmbeddingMismatchError(RuntimeError):
    """The collection was built with a different embedding backend than the current one."""


@dataclass
class StrategyIndexReport:
    strategy: str
    documents: int = 0
    chunks: int = 0
    seconds: float = 0.0


@dataclass
class IndexReport:
    embedding_backend: str = ""
    strategies: Dict[str, StrategyIndexReport] = field(default_factory=dict)
    documents: List[str] = field(default_factory=list)

    @property
    def total_chunks(self) -> int:
        return sum(r.chunks for r in self.strategies.values())


ProgressCallback = Callable[[float, str], None]


class VectorStore:
    """Thin, well-tested wrapper around a persistent ChromaDB client."""

    def __init__(self, persist_dir: Path | str | None = None, embedder: Optional[Embedder] = None) -> None:
        self.persist_dir = Path(persist_dir) if persist_dir is not None else config.CHROMA_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        try:
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        except Exception as exc:
            raise RuntimeError(
                f"Could not open the ChromaDB database at '{self.persist_dir}': {exc}. "
                "If the folder is corrupted, delete it and run 'python main.py --reset'."
            ) from exc

    # ------------------------------------------------------------------ basics
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    @staticmethod
    def collection_name(strategy: str) -> str:
        if strategy not in config.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(config.STRATEGIES)}")
        return f"{config.COLLECTION_PREFIX}_{strategy}"

    def close(self) -> None:
        """Release the client so the database folder can be reopened or deleted (Windows)."""
        try:
            self._client.clear_system_cache()
        except Exception:  # pragma: no cover - best effort
            pass
        self._client = None  # type: ignore[assignment]

    def _collection_metadata(self, strategy: str) -> Dict[str, Any]:
        return {
            "hnsw:space": config.DISTANCE_SPACE,
            "chunking_strategy": strategy,
            "embedding_backend": self.embedder.name,
            "embedding_dimension": int(self.embedder.dimension),
        }

    def has_collection(self, strategy: str) -> bool:
        try:
            self._client.get_collection(self.collection_name(strategy))
            return True
        except (NotFoundError, ValueError):
            return False

    def get_collection(self, strategy: str, create: bool = False):
        """Return the collection for ``strategy``.

        Raises CollectionNotFoundError when it does not exist and ``create`` is False.
        """
        name = self.collection_name(strategy)
        if create:
            return self._client.get_or_create_collection(
                name, metadata=self._collection_metadata(strategy), embedding_function=None
            )
        try:
            return self._client.get_collection(name, embedding_function=None)
        except (NotFoundError, ValueError) as exc:
            raise CollectionNotFoundError(
                f"No index found for strategy '{strategy}'. Run 'python main.py' (or use the "
                "Document Ingestion page in the app) to build it first."
            ) from exc

    def check_embedding_compatibility(self, collection) -> None:
        """Raise EmbeddingMismatchError if the collection came from another backend/dimension."""
        meta = collection.metadata or {}
        stored_backend = meta.get("embedding_backend")
        stored_dim = meta.get("embedding_dimension")
        if stored_dim is not None and int(stored_dim) != int(self.embedder.dimension):
            raise EmbeddingMismatchError(
                f"Collection '{collection.name}' stores {stored_dim}-dimensional vectors but the current "
                f"embedder produces {self.embedder.dimension}. Re-run 'python main.py --reset'."
            )
        if stored_backend is not None and stored_backend != self.embedder.name:
            raise EmbeddingMismatchError(
                f"Collection '{collection.name}' was built with '{stored_backend}' but the current embedder is "
                f"'{self.embedder.name}'. Vectors from different models cannot be compared. "
                "Re-run 'python main.py --reset' to rebuild the index."
            )

    # ------------------------------------------------------------------ writing
    def reset(self, strategy: Optional[str] = None) -> None:
        """Delete one strategy's collection, or all of them."""
        for item in ([strategy] if strategy else list(config.STRATEGIES)):
            try:
                self._client.delete_collection(self.collection_name(item))
                logger.info("Deleted collection for strategy '%s'", item)
            except (NotFoundError, ValueError):
                pass

    def add_chunks(self, strategy: str, chunks: Sequence[Chunk], extra_metadata: Optional[Dict[str, Any]] = None) -> int:
        """Embed and upsert chunks of ONE strategy. Returns the number stored.

        Any older chunks of the same documents are deleted first (idempotent re-ingestion).
        """
        chunks = [c for c in chunks if c.text.strip()]
        if not chunks:
            return 0
        wrong = {c.chunking_strategy for c in chunks} - {strategy}
        if wrong:
            raise ValueError(f"Chunks for strategies {sorted(wrong)} cannot be stored in the '{strategy}' collection")

        collection = self.get_collection(strategy, create=True)
        if collection.count() > 0:
            self.check_embedding_compatibility(collection)

        for document_id in sorted({c.document_id for c in chunks}):
            collection.delete(where={"document_id": document_id})

        stored = 0
        for start in range(0, len(chunks), UPSERT_BATCH):
            batch = chunks[start:start + UPSERT_BATCH]
            vectors = self.embedder.embed([c.text for c in batch])
            if vectors.shape != (len(batch), self.embedder.dimension):
                raise RuntimeError(f"Unexpected embedding shape {vectors.shape}")
            metadatas = []
            for chunk in batch:
                meta = chunk.metadata()
                meta.update(extra_metadata or {})
                metadatas.append({k: v for k, v in meta.items() if v is not None})
            collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=vectors.tolist(),
                documents=[c.text for c in batch],
                metadatas=metadatas,
            )
            stored += len(batch)
        return stored

    def index_documents(
        self,
        documents: Iterable[Document],
        strategies: Sequence[str] = config.STRATEGIES,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        progress: Optional[ProgressCallback] = None,
    ) -> IndexReport:
        """Chunk, embed and store documents for every requested strategy."""
        docs = list(documents)
        report = IndexReport(embedding_backend=self.embedder.name, documents=[d.document_name for d in docs])
        steps = max(1, len(docs) * len(strategies))
        done = 0
        for strategy in strategies:
            # a collection built by a different embedder must be rebuilt, never mixed
            if self.has_collection(strategy):
                existing = self.get_collection(strategy)
                try:
                    self.check_embedding_compatibility(existing)
                except EmbeddingMismatchError as exc:
                    logger.warning("%s Rebuilding this collection.", exc)
                    self.reset(strategy)
            entry = StrategyIndexReport(strategy=strategy)
            started = time.perf_counter()
            for doc in docs:
                if progress:
                    progress(done / steps, f"{doc.document_name} [{strategy}]")
                chunks = chunk_document(doc, strategy, chunk_size, chunk_overlap)
                extra = {
                    "content_hash": doc.content_hash,
                    "chunk_size_setting": chunk_size,
                    "chunk_overlap_setting": chunk_overlap,
                    "document_total_pages": doc.total_pages,
                }
                entry.chunks += self.add_chunks(strategy, chunks, extra)
                entry.documents += 1
                done += 1
            entry.seconds = time.perf_counter() - started
            report.strategies[strategy] = entry
            logger.info("Indexed %d chunk(s) from %d document(s) for '%s' in %.2fs",
                        entry.chunks, entry.documents, strategy, entry.seconds)
        if progress:
            progress(1.0, "done")
        return report

    # ------------------------------------------------------------------ reading
    def count(self, strategy: str) -> int:
        try:
            return self.get_collection(strategy).count()
        except CollectionNotFoundError:
            return 0

    def query(self, strategy: str, query_embedding: Sequence[float], top_k: int) -> Dict[str, Any]:
        """Nearest-neighbour search. Returns Chroma's raw result (ids, documents, metadatas, distances)."""
        collection = self.get_collection(strategy)
        self.check_embedding_compatibility(collection)
        available = collection.count()
        if available == 0:
            raise CollectionNotFoundError(f"The '{strategy}' index exists but is empty. Run ingestion first.")
        return collection.query(
            query_embeddings=[list(map(float, query_embedding))],
            n_results=min(top_k, available),
            include=["documents", "metadatas", "distances"],
        )

    def get_all_metadata(self, strategy: str) -> List[Dict[str, Any]]:
        try:
            collection = self.get_collection(strategy)
        except CollectionNotFoundError:
            return []
        return list(collection.get(include=["metadatas"]).get("metadatas") or [])

    def stats(self) -> Dict[str, Any]:
        """Numbers for the dashboard."""
        chunks = {s: self.count(s) for s in config.STRATEGIES}
        documents: Dict[str, int] = {}  # document_id -> total pages in the PDF
        pages: set[tuple[str, int]] = set()
        backend = ""
        for strategy in config.STRATEGIES:
            for meta in self.get_all_metadata(strategy):
                doc_id = str(meta.get("document_id"))
                documents[doc_id] = int(meta.get("document_total_pages", 0) or 0)
                pages.add((doc_id, int(meta.get("page_number", 0))))
            if self.has_collection(strategy) and not backend:
                backend = str((self.get_collection(strategy).metadata or {}).get("embedding_backend", ""))
        return {
            "documents": len(documents),
            "document_ids": sorted(documents),
            "pages_with_text": len(pages),
            "total_pdf_pages": sum(documents.values()),
            "chunks": chunks,
            "total_chunks": sum(chunks.values()),
            "embeddings": sum(chunks.values()),  # one stored vector per chunk
            "embedding_backend": backend,
            "persist_dir": str(self.persist_dir),
        }
