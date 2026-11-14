"""Text embeddings.

Default backend: ``sentence-transformers`` with ``all-MiniLM-L6-v2`` (384 dimensions).
The model is downloaded once into ``models_cache/`` (or reused from the normal Hugging
Face cache) and kept in memory for the life of the process.

Offline fallback backend: ``hashing`` - a small, deterministic, *lexical* embedder
(hashed word and bigram counts) that also outputs 384 dimensions. It exists so that the
pipeline, tests and reports can run on a machine that cannot download the model. It is
much less capable than MiniLM (it matches words, not meaning), so every database and
report records which backend produced it, and the app shows a warning when it is active.
"""

from __future__ import annotations

import math
import re
import threading
import zlib
from typing import List, Optional, Protocol, Sequence

import numpy as np

import config
from src.utils import get_logger

logger = get_logger(__name__)

BACKEND_ST = "sentence-transformers"
BACKEND_HASHING = "hashing"


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced."""


class Embedder(Protocol):
    name: str  # e.g. "sentence-transformers:all-MiniLM-L6-v2"
    backend: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def _check_texts(texts: Sequence[str]) -> List[str]:
    if isinstance(texts, str):
        raise TypeError("embed() expects a list of strings, not a single string")
    cleaned = [t if isinstance(t, str) else str(t) for t in texts]
    return cleaned


# --------------------------------------------------------------------------- real model
class SentenceTransformerEmbedder:
    """Wrapper around a sentence-transformers model (cosine-ready, L2-normalised)."""

    backend = BACKEND_ST

    def __init__(self, model_name: str = config.EMBEDDING_MODEL, local_only: bool = False) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise EmbeddingError("sentence-transformers is not installed (pip install -r requirements.txt)") from exc
        config.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            self._model = SentenceTransformer(
                model_name, cache_folder=str(config.MODEL_CACHE_DIR), local_files_only=local_only
            )
        except Exception as exc:
            raise EmbeddingError(f"Could not load embedding model '{model_name}': {exc}") from exc
        self.model_name = model_name
        self.name = f"{BACKEND_ST}:{model_name}"
        dim = self._model.get_sentence_embedding_dimension()
        self.dimension = int(dim) if dim else config.EMBEDDING_DIMENSION

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        items = _check_texts(texts)
        if not items:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            items,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


# --------------------------------------------------------------------------- fallback
_STOPWORDS = frozenset(
    "a an the and or but if then of to in on at by for with from as is are was were be been being it its this that "
    "these those which who whom what when where why how can could may might will would should do does did done not no "
    "than so such also into over under between about after before more most other some any each both their there they "
    "them we our you your he she his her i me my have has had having use used using often one two".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    """Very light suffix stripping so 'predictions' and 'predict' share a feature."""
    for suffix in ("ations", "ation", "ingly", "ments", "ment", "ities", "ity", "ings", "ing", "ies", "ied", "ed", "es", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


class HashingEmbedder:
    """Deterministic lexical embedder (offline fallback, NOT semantic like MiniLM)."""

    backend = BACKEND_HASHING

    def __init__(self, dimension: int = config.EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension
        self.name = f"{BACKEND_HASHING}:{dimension}d"

    def _features(self, text: str) -> List[str]:
        words = [_stem(w) for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def _embed_one(self, text: str) -> np.ndarray:
        counts: dict[str, int] = {}
        for feature in self._features(text):
            counts[feature] = counts.get(feature, 0) + 1
        vector = np.zeros(self.dimension, dtype=np.float32)
        for feature, count in counts.items():
            digest = zlib.crc32(feature.encode("utf-8"))
            index = digest % self.dimension
            sign = 1.0 if (digest >> 16) & 1 else -1.0
            weight = (1.0 + math.log(count)) * (0.6 if "_" in feature else 1.0)
            vector[index] += sign * weight
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        items = _check_texts(texts)
        if not items:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._embed_one(t) for t in items]).astype(np.float32)


# --------------------------------------------------------------------------- factory
_lock = threading.Lock()
_cache: dict[str, Embedder] = {}


def _load_sentence_transformer() -> SentenceTransformerEmbedder:
    """Prefer the local cache; only touch the network if the model is not cached yet."""
    try:
        return SentenceTransformerEmbedder(config.EMBEDDING_MODEL, local_only=True)
    except EmbeddingError:
        logger.info("Model '%s' not cached locally; downloading once...", config.EMBEDDING_MODEL)
        return SentenceTransformerEmbedder(config.EMBEDDING_MODEL, local_only=False)


def get_embedder(backend: Optional[str] = None) -> Embedder:
    """Return a cached embedder (the model is loaded once per process).

    ``backend``: "auto" (default from config), "sentence-transformers" or "hashing".
    In "auto" mode a failure to load the real model falls back to the hashing embedder
    with a loud warning, unless ``config.ALLOW_FALLBACK_EMBEDDER`` is false.
    """
    requested = (backend or config.EMBEDDING_BACKEND).lower()
    if requested not in ("auto", BACKEND_ST, BACKEND_HASHING):
        raise ValueError(f"Unknown embedding backend '{requested}'")
    with _lock:
        if requested in _cache:
            return _cache[requested]
        if requested == BACKEND_HASHING:
            embedder: Embedder = HashingEmbedder()
        elif requested == BACKEND_ST:
            embedder = _load_sentence_transformer()
        else:  # auto
            try:
                embedder = _load_sentence_transformer()
            except EmbeddingError as exc:
                if not config.ALLOW_FALLBACK_EMBEDDER:
                    raise
                logger.warning(
                    "!! Could not load '%s' (%s). Using the OFFLINE HASHING FALLBACK embedder instead. "
                    "Results will be lexical, not semantic. Connect to the internet once and re-run "
                    "'python main.py --reset' to use the real model.",
                    config.EMBEDDING_MODEL, exc,
                )
                embedder = HashingEmbedder()
        _cache[requested] = embedder
        logger.info("Embedding backend ready: %s (%d dimensions)", embedder.name, embedder.dimension)
        return embedder


def embed_texts(texts: Sequence[str], embedder: Optional[Embedder] = None) -> np.ndarray:
    """Embed a list of texts with the given (or default) embedder."""
    return (embedder or get_embedder()).embed(texts)


def reset_embedder_cache() -> None:
    """Forget cached embedders (used by tests)."""
    with _lock:
        _cache.clear()
