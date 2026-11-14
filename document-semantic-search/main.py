"""Command-line entry point: ingest the sample PDFs, verify persistence, run the evaluation.

    python main.py              # ingest + evaluate (safe to repeat; no duplicates)
    python main.py --reset      # wipe the database first and rebuild from scratch
    python main.py --skip-eval  # ingest only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config
from src.document_loader import list_pdf_files
from src.embeddings import get_embedder
from src.evaluation import run_evaluation, write_reports
from src.pipeline import ingest
from src.utils import get_logger
from src.vector_store import VectorStore

logger = get_logger("main")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Document ingestion, chunking and semantic search pipeline")
    parser.add_argument("--reset", action="store_true", help="delete the existing ChromaDB collections first")
    parser.add_argument("--skip-eval", action="store_true", help="only ingest; do not run the evaluation")
    parser.add_argument("--documents-dir", type=Path, default=None, help="folder with PDFs (default: data/documents)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    docs_dir = args.documents_dir or config.DOCUMENTS_DIR

    try:
        pdfs = list_pdf_files(docs_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    if not pdfs:
        logger.error("No PDF files found in %s. Add at least one PDF and run again.", docs_dir)
        return 1
    logger.info("Found %d PDF file(s) in %s", len(pdfs), docs_dir)

    try:
        embedder = get_embedder()
    except Exception as exc:
        logger.error("Could not prepare the embedding model: %s", exc)
        return 1
    logger.info("Embedding backend: %s (%d dimensions)", embedder.name, embedder.dimension)

    store = VectorStore(config.CHROMA_DIR, embedder)
    try:
        report, failures = ingest(store, directory=docs_dir, reset=args.reset)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("Ingestion failed: %s", exc)
        return 1
    for failure in failures:
        logger.warning("Skipped %s: %s", Path(failure.path).name, failure.reason)

    expected = {s: r.chunks for s, r in report.strategies.items()}
    store.close()  # close and reopen to prove the data persisted on disk
    store = VectorStore(config.CHROMA_DIR, embedder)
    reopened = {s: store.count(s) for s in config.STRATEGIES}
    if reopened != expected:
        logger.error("Persistence check FAILED: wrote %s but found %s after reopening", expected, reopened)
        return 1
    logger.info("Persistence check passed: %s chunks found after reopening the database", reopened)

    stats = store.stats()
    print("\n=== Ingestion summary ===")
    print(f"documents ingested : {stats['documents']} ({stats['pages_with_text']} pages with text)")
    for strategy, count in stats["chunks"].items():
        print(f"chunks [{strategy:<9}] : {count}")
    print(f"stored embeddings  : {stats['embeddings']}  (backend: {stats['embedding_backend']})")

    if args.skip_eval:
        return 0

    result = run_evaluation(store=store, embedder=embedder)
    paths = write_reports(result)
    print("\n=== Evaluation summary (mean over the 3 queries) ===")
    print(result.overall.to_string(index=False))
    print(f"\nCSV report      : {paths['csv']}")
    print(f"Markdown report : {paths['markdown']}")
    if result.embedding_backend.startswith("hashing"):
        print("\nNOTE: the offline hashing fallback embedder was used. Re-run with internet access "
              "(python main.py --reset) to use all-MiniLM-L6-v2.")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
