"""Small shared helpers: logging, slugs, hashing and similarity conversion."""

from __future__ import annotations

import hashlib
import logging
import re
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure console logging once (safe to call repeatedly)."""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    # Quieten very chatty third-party libraries
    for noisy in ("httpx", "urllib3", "chromadb.telemetry", "sentence_transformers", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def slugify(text: str) -> str:
    """Turn a file stem into a safe identifier: 'My Doc (v2)' -> 'my_doc_v2'."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "document"


def file_sha256(path: Path, block_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes; used to detect that a document changed."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def distance_to_similarity(distance: float) -> float:
    """Convert a ChromaDB *cosine distance* into a cosine *similarity*.

    Chroma's cosine space reports ``distance = 1 - cosine_similarity`` (0 = identical
    direction, 1 = unrelated, 2 = opposite). The similarity is therefore
    ``1 - distance`` and lies in [-1, 1]. It is a cosine score, NOT a percentage and
    NOT a probability. We clamp tiny floating-point overshoots to the valid range.
    """
    return max(-1.0, min(1.0, 1.0 - float(distance)))


def clean_text(text: str) -> str:
    """Normalise whitespace in extracted PDF text while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
