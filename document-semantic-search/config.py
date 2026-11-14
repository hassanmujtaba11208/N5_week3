"""Central configuration for the Document Semantic Search pipeline.

Every tunable value lives here. Most of them can also be overridden with an
environment variable, which is handy for tests and for quick experiments, e.g. in
PowerShell:   $env:CHUNK_SIZE = "600"; python main.py
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DOCUMENTS_DIR: Path = Path(os.getenv("DOCUMENTS_DIR", PROJECT_ROOT / "data" / "documents"))
PROCESSED_DIR: Path = Path(os.getenv("PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))
CHROMA_DIR: Path = Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / "database" / "chroma_db"))
REPORTS_DIR: Path = Path(os.getenv("REPORTS_DIR", PROJECT_ROOT / "reports"))
MODEL_CACHE_DIR: Path = Path(os.getenv("MODEL_CACHE_DIR", PROJECT_ROOT / "models_cache"))

CSV_REPORT_PATH: Path = REPORTS_DIR / "chunking_comparison.csv"
MD_REPORT_PATH: Path = REPORTS_DIR / "evaluation_report.md"

# --------------------------------------------------------------------------- embeddings
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION: int = 384  # output size of all-MiniLM-L6-v2 (the fallback matches it)
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# If the sentence-transformers model cannot be loaded (for example no internet on the
# very first run), the pipeline can fall back to a small offline "hashing" embedder so
# that everything still runs. The backend that was really used is ALWAYS recorded in the
# database and printed in the reports. Set to "0" to make a missing model a hard error.
ALLOW_FALLBACK_EMBEDDER: bool = os.getenv("ALLOW_FALLBACK_EMBEDDER", "1") == "1"
# Force a backend: "auto" (default), "sentence-transformers" or "hashing".
EMBEDDING_BACKEND: str = os.getenv("EMBEDDING_BACKEND", "auto")

# --------------------------------------------------------------------------- chunking
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))  # characters
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))  # characters
RECURSIVE_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")
MIN_CHUNK_CHARS: int = int(os.getenv("MIN_CHUNK_CHARS", "30"))  # drop fragments shorter than this

STRATEGY_FIXED: str = "fixed"
STRATEGY_RECURSIVE: str = "recursive"
STRATEGIES: tuple[str, ...] = (STRATEGY_FIXED, STRATEGY_RECURSIVE)

# --------------------------------------------------------------------------- search
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
MAX_TOP_K: int = 50
MAX_QUERY_CHARS: int = 2000

# ChromaDB collections (one per strategy so they stay separately identifiable)
COLLECTION_PREFIX: str = "docsearch"
DISTANCE_SPACE: str = "cosine"  # Chroma returns cosine DISTANCE = 1 - cosine similarity

# --------------------------------------------------------------------------- evaluation
EVAL_TOP_K: int = 5
# (query, keywords a genuinely relevant passage is expected to mention)
EVAL_QUERIES: list[dict] = [
    {
        "query": "What are the main benefits of artificial intelligence in healthcare?",
        "keywords": ["healthcare", "diagnos", "patient", "clinician", "detect", "treatment", "hospital", "benefit"],
    },
    {
        "query": "How does machine learning use training data to make predictions?",
        "keywords": ["training", "data", "model", "predict", "label", "feature", "parameter", "inference"],
    },
    {
        "query": "What are the main challenges of data security and privacy?",
        "keywords": ["security", "privacy", "breach", "attack", "personal data", "encrypt", "access", "regulat"],
    },
]
