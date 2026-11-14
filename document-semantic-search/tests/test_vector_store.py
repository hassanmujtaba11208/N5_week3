"""Tests for ChromaDB insertion, metadata, duplicate prevention and persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

import config
from src.chunking import chunk_document
from src.document_loader import load_documents
from src.embeddings import HashingEmbedder
from src.pipeline import ingest
from src.vector_store import CollectionNotFoundError, EmbeddingMismatchError, VectorStore
from tests.conftest import PAGE_TEXTS, make_pdf

REQUIRED_METADATA = {
    "document_name", "document_id", "page_number", "chunk_id", "chunking_strategy",
    "chunk_index", "source_file",
}


def test_insertion_stores_chunks_for_both_strategies(indexed_store: VectorStore):
    fixed, recursive = indexed_store.count("fixed"), indexed_store.count("recursive")
    assert fixed > 0 and recursive > 0
    assert indexed_store.stats()["embeddings"] == fixed + recursive


def test_metadata_is_preserved(indexed_store: VectorStore):
    for strategy in config.STRATEGIES:
        collection = indexed_store.get_collection(strategy)
        data = collection.get(include=["metadatas", "documents"])
        assert data["ids"]
        for chunk_id, meta, text in zip(data["ids"], data["metadatas"], data["documents"]):
            assert REQUIRED_METADATA <= set(meta)
            assert meta["chunk_id"] == chunk_id
            assert meta["chunking_strategy"] == strategy
            assert meta["document_name"].endswith(".pdf") and meta["page_number"] >= 1
            assert text.strip()


def test_strategies_live_in_separate_collections(indexed_store: VectorStore):
    assert indexed_store.collection_name("fixed") != indexed_store.collection_name("recursive")
    for strategy in config.STRATEGIES:
        meta = indexed_store.get_collection(strategy).metadata
        assert meta["chunking_strategy"] == strategy and meta["hnsw:space"] == "cosine"
        ids = indexed_store.get_collection(strategy).get()["ids"]
        assert all(f"__{strategy}__" in i for i in ids)


def test_reingestion_does_not_create_duplicates(store: VectorStore, docs_dir: Path):
    ingest(store, directory=docs_dir, chunk_size=300, chunk_overlap=60)
    first = {s: store.count(s) for s in config.STRATEGIES}
    ids_first = {s: set(store.get_collection(s).get()["ids"]) for s in config.STRATEGIES}
    for _ in range(2):
        ingest(store, directory=docs_dir, chunk_size=300, chunk_overlap=60)
    assert {s: store.count(s) for s in config.STRATEGIES} == first
    assert {s: set(store.get_collection(s).get()["ids"]) for s in config.STRATEGIES} == ids_first


def test_changed_document_replaces_old_chunks(store: VectorStore, tmp_path: Path):
    folder = tmp_path / "docs"
    folder.mkdir()
    pdf = make_pdf(folder / "report.pdf", PAGE_TEXTS["ml_doc"])
    ingest(store, directory=folder, chunk_size=300, chunk_overlap=60)
    before = store.count("recursive")
    make_pdf(pdf, ["Completely new single page about quantum gardening and tulips, enough words to make a chunk."])
    ingest(store, directory=folder, chunk_size=300, chunk_overlap=60)
    after = store.get_collection("recursive").get(include=["documents"])
    assert store.count("recursive") < before
    assert all("quantum gardening" in d for d in after["documents"])


def test_database_persists_after_close_and_reopen(tmp_path: Path, docs_dir: Path):
    path = tmp_path / "persist_db"
    first = VectorStore(path, HashingEmbedder())
    ingest(first, directory=docs_dir, chunk_size=300, chunk_overlap=60)
    counts = {s: first.count(s) for s in config.STRATEGIES}
    sample = first.get_collection("recursive").get(limit=1, include=["metadatas", "documents"])
    first.close()

    reopened = VectorStore(path, HashingEmbedder())
    assert {s: reopened.count(s) for s in config.STRATEGIES} == counts
    again = reopened.get_collection("recursive").get(ids=sample["ids"], include=["metadatas", "documents"])
    assert again["documents"] == sample["documents"] and again["metadatas"] == sample["metadatas"]
    reopened.close()


def test_missing_collection_raises_clear_error(store: VectorStore):
    assert store.count("fixed") == 0
    with pytest.raises(CollectionNotFoundError):
        store.get_collection("fixed")
    with pytest.raises(CollectionNotFoundError):
        store.query("recursive", [0.0] * config.EMBEDDING_DIMENSION, 3)


def test_unknown_strategy_rejected(store: VectorStore):
    with pytest.raises(ValueError):
        store.get_collection("semantic")


def test_chunks_for_wrong_strategy_are_rejected(store: VectorStore, docs_dir: Path):
    docs, _ = load_documents(directory=docs_dir)
    chunks = chunk_document(docs[0], "fixed", 300, 60)
    with pytest.raises(ValueError):
        store.add_chunks("recursive", chunks)


def test_embedding_backend_mismatch_is_detected(indexed_store: VectorStore):
    class OtherEmbedder(HashingEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.name = "other-model:384d"

    other = VectorStore(indexed_store.persist_dir, OtherEmbedder())
    with pytest.raises(EmbeddingMismatchError):
        other.query("fixed", [0.1] * config.EMBEDDING_DIMENSION, 3)
    other.close()


def test_reset_removes_collections(indexed_store: VectorStore):
    indexed_store.reset("fixed")
    assert indexed_store.count("fixed") == 0 and indexed_store.count("recursive") > 0
    indexed_store.reset()
    assert indexed_store.stats()["total_chunks"] == 0


def test_stats_report_documents_and_pages(indexed_store: VectorStore):
    stats = indexed_store.stats()
    assert stats["documents"] == 3
    assert stats["pages_with_text"] == 9 and stats["total_pdf_pages"] == 9
    assert stats["embedding_backend"].startswith("hashing")


def test_reingest_with_a_different_embedder_rebuilds_instead_of_mixing(indexed_store: VectorStore, docs_dir: Path):
    class OtherEmbedder(HashingEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.name = "other-model:384d"

    other = VectorStore(indexed_store.persist_dir, OtherEmbedder())
    ingest(other, directory=docs_dir, chunk_size=300, chunk_overlap=60)
    for strategy in config.STRATEGIES:
        assert other.get_collection(strategy).metadata["embedding_backend"] == "other-model:384d"
        assert other.count(strategy) > 0
    assert other.stats()["embedding_backend"] == "other-model:384d"
    other.close()
