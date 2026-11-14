"""Tests for the search() function: ranking, scores, top-k limits, errors."""

from __future__ import annotations

import inspect

import pytest

import config
from src import search as search_module
from src.search import InvalidQueryError, search, timed_search
from src.vector_store import CollectionNotFoundError, VectorStore

HIT_KEYS = {
    "rank", "similarity", "distance", "document_name", "document_id", "page_number",
    "chunk_id", "chunk_index", "chunking_strategy", "source_file", "text",
}


def test_signature_matches_assignment():
    params = inspect.signature(search).parameters
    assert list(params)[:3] == ["query", "top_k", "strategy"]
    assert params["top_k"].default == 5 and params["strategy"].default == "recursive"


@pytest.mark.parametrize("strategy", config.STRATEGIES)
def test_results_have_all_fields_and_are_ranked(indexed_store: VectorStore, strategy: str):
    hits = search("how do models learn from training data", 5, strategy, store=indexed_store)
    assert 1 <= len(hits) <= 5
    for hit in hits:
        assert HIT_KEYS <= set(hit)
        assert hit["chunking_strategy"] == strategy
        assert hit["text"].strip() and hit["document_name"].endswith(".pdf") and hit["page_number"] >= 1
    assert [h["rank"] for h in hits] == list(range(1, len(hits) + 1))
    sims = [h["similarity"] for h in hits]
    assert sims == sorted(sims, reverse=True)
    dists = [h["distance"] for h in hits]
    assert dists == sorted(dists)


def test_similarity_is_one_minus_distance_not_a_percentage(indexed_store: VectorStore):
    for hit in search("encryption protects personal data", 5, "recursive", store=indexed_store):
        assert hit["similarity"] == pytest.approx(1.0 - hit["distance"], abs=2e-4)
        assert -1.0 <= hit["similarity"] <= 1.0


@pytest.mark.parametrize("top_k", [1, 2, 3, 7])
def test_top_k_limits_number_of_results(indexed_store: VectorStore, top_k: int):
    for strategy in config.STRATEGIES:
        assert len(search("patients and hospitals", top_k, strategy, store=indexed_store)) == min(top_k, indexed_store.count(strategy))


def test_top_k_larger_than_index_returns_everything(indexed_store: VectorStore):
    total = indexed_store.count("fixed")
    assert len(search("data", config.MAX_TOP_K, "fixed", store=indexed_store)) == min(config.MAX_TOP_K, total)


def test_results_are_topically_relevant(indexed_store: VectorStore):
    top = {
        "How do doctors and hospitals use AI to detect disease in patients?": "health_doc.pdf",
        "How does a model learn from training data and make predictions?": "ml_doc.pdf",
        "How do encryption and access control prevent a data breach?": "security_doc.pdf",
    }
    for query, expected in top.items():
        for strategy in config.STRATEGIES:
            hits = search(query, 3, strategy, store=indexed_store)
            assert hits[0]["document_name"] == expected, (query, strategy, hits[0]["document_name"])


def test_default_strategy_is_recursive(indexed_store: VectorStore):
    hits = search("training data", store=indexed_store)
    assert {h["chunking_strategy"] for h in hits} == {"recursive"} and len(hits) <= 5


def test_whitespace_in_query_is_normalised(indexed_store: VectorStore):
    a = search("training   data\n for models", 3, "fixed", store=indexed_store)
    b = search("training data for models", 3, "fixed", store=indexed_store)
    assert [h["chunk_id"] for h in a] == [h["chunk_id"] for h in b]


@pytest.mark.parametrize("bad_query", ["", "   ", "\n\t", None, 123, ["a list"]])
def test_invalid_queries_raise(indexed_store: VectorStore, bad_query):
    with pytest.raises(InvalidQueryError):
        search(bad_query, 3, "fixed", store=indexed_store)  # type: ignore[arg-type]


def test_overlong_query_rejected(indexed_store: VectorStore):
    with pytest.raises(InvalidQueryError):
        search("word " * 1000, 3, "fixed", store=indexed_store)


@pytest.mark.parametrize("bad_k", [0, -1, config.MAX_TOP_K + 1, 2.5, "3", None, True])
def test_invalid_top_k_raises(indexed_store: VectorStore, bad_k):
    with pytest.raises(InvalidQueryError):
        search("data", bad_k, "fixed", store=indexed_store)  # type: ignore[arg-type]


def test_invalid_strategy_raises(indexed_store: VectorStore):
    with pytest.raises(InvalidQueryError):
        search("data", 3, "semantic", store=indexed_store)


def test_missing_collection_is_reported_clearly(store: VectorStore):
    with pytest.raises(CollectionNotFoundError, match="python main.py"):
        search("anything at all", 3, "recursive", store=store)


def test_one_strategy_can_be_missing(store: VectorStore, docs_dir):
    from src.pipeline import ingest

    ingest(store, directory=docs_dir, strategies=["fixed"], chunk_size=300, chunk_overlap=60)
    assert search("training data", 2, "fixed", store=store)
    with pytest.raises(CollectionNotFoundError):
        search("training data", 2, "recursive", store=store)


def test_timed_search_returns_milliseconds(indexed_store: VectorStore):
    hits, ms = timed_search("hospital patients", 3, "recursive", store=indexed_store)
    assert hits and 0 < ms < 60_000


def test_default_store_is_used_when_none_given(indexed_store: VectorStore, monkeypatch):
    monkeypatch.setattr(search_module, "get_default_store", lambda: indexed_store)
    assert search("training data", 2, "fixed")
