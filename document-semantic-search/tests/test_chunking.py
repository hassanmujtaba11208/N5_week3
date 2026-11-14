"""Tests for both chunking strategies."""

from __future__ import annotations

from pathlib import Path

import pytest

import config
from src.chunking import chunk_document, chunk_documents, fixed_size_split, recursive_split
from src.document_loader import Document, load_documents
from src.text_extractor import PageText

SENTENCE = "This is a reasonably long sentence about machine learning models and their data."
PARAGRAPHS = "\n\n".join(" ".join([SENTENCE] * 3) for _ in range(6))  # six paragraphs of ~250 chars


def _document(text: str, pages: int = 1) -> Document:
    return Document(
        document_id="demo", document_name="demo.pdf", source_file="data/documents/demo.pdf",
        content_hash="x" * 64, total_pages=pages, pages=[PageText(i + 1, text) for i in range(pages)],
    )


# ----------------------------------------------------------------------- fixed-size
def test_fixed_chunks_respect_size_and_cover_text():
    text = "abcdefghij" * 100  # 1000 chars
    chunks = fixed_size_split(text, chunk_size=200, chunk_overlap=50)
    assert all(0 < len(c) <= 200 for c in chunks)
    assert len(chunks) == 7  # windows start every 150 chars: 0,150,...,900 (last reaches the end)
    assert chunks[0].startswith("abcdefghij") and text.endswith(chunks[-1])


def test_fixed_chunks_overlap_by_configured_amount():
    text = "".join(chr(97 + (i % 26)) for i in range(600))
    chunks = fixed_size_split(text, chunk_size=200, chunk_overlap=50)
    for first, second in zip(chunks, chunks[1:]):
        assert first[-50:] == second[:50]


def test_fixed_zero_overlap_has_no_repetition():
    chunks = fixed_size_split("x" * 100 + "y" * 100, chunk_size=100, chunk_overlap=0)
    assert chunks == ["x" * 100, "y" * 100]


def test_fixed_ignores_sentence_boundaries():
    chunks = fixed_size_split(PARAGRAPHS, chunk_size=200, chunk_overlap=0)
    assert any(not c.rstrip().endswith((".", "!", "?")) for c in chunks)  # cuts through sentences


@pytest.mark.parametrize("size,overlap", [(0, 0), (-5, 0), (100, -1), (100, 100), (100, 150)])
def test_invalid_parameters_raise(size: int, overlap: int):
    with pytest.raises(ValueError):
        fixed_size_split("some text", size, overlap)
    with pytest.raises(ValueError):
        recursive_split("some text", size, overlap)


@pytest.mark.parametrize("splitter", [fixed_size_split, recursive_split])
@pytest.mark.parametrize("text", ["", "   ", "\n\n\n", "\t \n"])
def test_empty_text_gives_no_chunks(splitter, text: str):
    assert splitter(text, 200, 20) == []


@pytest.mark.parametrize("splitter", [fixed_size_split, recursive_split])
def test_short_text_is_a_single_chunk(splitter):
    assert splitter("A short paragraph.", 200, 20) == ["A short paragraph."]


# ----------------------------------------------------------------------- recursive
def test_recursive_chunks_respect_size_and_have_no_empty_chunks():
    chunks = recursive_split(PARAGRAPHS, chunk_size=400, chunk_overlap=80)
    assert len(chunks) > 1
    assert all(c.strip() and len(c) <= 400 for c in chunks)


def test_recursive_keeps_sentences_intact():
    chunks = recursive_split(PARAGRAPHS, chunk_size=400, chunk_overlap=80)
    # every sentence is short (<400) so the recursive splitter never needs to cut one in half
    for chunk in chunks:
        assert chunk.rstrip().endswith("."), chunk[-40:]
        assert chunk.startswith("This is"), chunk[:40]


def test_recursive_prefers_paragraph_boundaries():
    paragraphs = ["First paragraph " + "alpha " * 30, "Second paragraph " + "beta " * 30, "Third paragraph " + "gamma " * 30]
    text = "\n\n".join(p.strip() for p in paragraphs)
    chunks = recursive_split(text, chunk_size=230, chunk_overlap=0)
    assert len(chunks) == 3
    assert chunks[0].startswith("First") and chunks[1].startswith("Second") and chunks[2].startswith("Third")


def test_recursive_overlap_repeats_context():
    chunks = recursive_split(PARAGRAPHS, chunk_size=520, chunk_overlap=260)
    shared = [a.split(".")[-2] in b for a, b in zip(chunks, chunks[1:]) if "." in a]
    assert any(shared)


def test_recursive_hard_splits_text_without_separators():
    chunks = recursive_split("x" * 1000, chunk_size=300, chunk_overlap=30)
    assert all(len(c) <= 300 for c in chunks) and len(chunks) >= 4


def test_strategies_are_genuinely_different():
    fixed = fixed_size_split(PARAGRAPHS, 400, 80)
    recursive = recursive_split(PARAGRAPHS, 400, 80)
    assert fixed != recursive


# ----------------------------------------------------------------------- documents
@pytest.mark.parametrize("strategy", config.STRATEGIES)
def test_chunk_document_ids_and_metadata(strategy: str):
    doc = _document(PARAGRAPHS, pages=3)
    chunks = chunk_document(doc, strategy, 300, 50)
    assert chunks
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))  # unique
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.page_number for c in chunks} == {1, 2, 3}
    for chunk in chunks:
        meta = chunk.metadata()
        assert meta["document_name"] == "demo.pdf" and meta["document_id"] == "demo"
        assert meta["chunking_strategy"] == strategy and meta["source_file"] == "data/documents/demo.pdf"
        assert meta["chunk_id"] == chunk.chunk_id and "text" not in meta
        assert strategy in chunk.chunk_id and f"p{chunk.page_number:03d}" in chunk.chunk_id
        assert chunk.text.strip()


@pytest.mark.parametrize("strategy", config.STRATEGIES)
def test_chunk_ids_are_deterministic(strategy: str):
    doc = _document(PARAGRAPHS)
    assert [c.chunk_id for c in chunk_document(doc, strategy, 300, 50)] == [c.chunk_id for c in chunk_document(doc, strategy, 300, 50)]


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_document(_document("text " * 50), "semantic", 200, 20)


def test_document_with_only_whitespace_pages_gives_no_chunks():
    doc = Document("e", "e.pdf", "e.pdf", "0" * 64, 1, [PageText(1, "   \n  ")])
    for strategy in config.STRATEGIES:
        assert chunk_document(doc, strategy, 200, 20) == []


def test_both_strategies_work_on_same_documents(docs_dir: Path):
    documents, _ = load_documents(directory=docs_dir)
    fixed = chunk_documents(documents, config.STRATEGY_FIXED, 300, 60)
    recursive = chunk_documents(documents, config.STRATEGY_RECURSIVE, 300, 60)
    assert fixed and recursive
    assert {c.document_name for c in fixed} == {c.document_name for c in recursive} == {d.document_name for d in documents}
