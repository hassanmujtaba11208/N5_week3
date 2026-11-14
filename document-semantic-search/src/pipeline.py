"""High-level ingestion workflow shared by main.py and the Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import config
from src.document_loader import LoadFailure, load_documents
from src.utils import get_logger
from src.vector_store import IndexReport, ProgressCallback, VectorStore

logger = get_logger(__name__)


def ingest(
    store: VectorStore,
    paths: Optional[Iterable[Path | str]] = None,
    directory: Path | str | None = None,
    reset: bool = False,
    strategies: Sequence[str] = config.STRATEGIES,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    progress: Optional[ProgressCallback] = None,
) -> Tuple[IndexReport, List[LoadFailure]]:
    """Load PDFs, chunk them with every strategy, embed and store them.

    Safe to run repeatedly: existing chunks of the same documents are replaced, never duplicated.
    Raises FileNotFoundError if the documents folder is missing and ValueError if no
    document could be loaded at all.
    """
    if reset:
        store.reset()
    documents, failures = load_documents(paths, directory)
    if not documents:
        reasons = "; ".join(f"{Path(f.path).name}: {f.reason}" for f in failures) or "no PDF files found"
        raise ValueError(f"No documents could be ingested ({reasons})")
    report = store.index_documents(documents, strategies, chunk_size, chunk_overlap, progress)
    return report, failures
