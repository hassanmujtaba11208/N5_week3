"""Tests for embedding generation."""

from __future__ import annotations

import numpy as np
import pytest

import config
from src import embeddings
from src.embeddings import EmbeddingError, HashingEmbedder, SentenceTransformerEmbedder, get_embedder, reset_embedder_cache


def test_hashing_embeddings_have_expected_shape_and_are_normalised(embedder: HashingEmbedder):
    vectors = embedder.embed(["machine learning uses data", "hospital patients and doctors"])
    assert vectors.shape == (2, config.EMBEDDING_DIMENSION)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_embeddings_are_deterministic(embedder: HashingEmbedder):
    a = embedder.embed(["the same sentence"])
    b = HashingEmbedder().embed(["the same sentence"])
    assert np.array_equal(a, b)


def test_related_texts_are_closer_than_unrelated(embedder: HashingEmbedder):
    q, related, unrelated = embedder.embed([
        "training data for machine learning models",
        "a model is trained on labelled training data",
        "encryption keys protect stored files",
    ])
    assert float(q @ related) > float(q @ unrelated)


def test_empty_list_and_empty_string(embedder: HashingEmbedder):
    assert embedder.embed([]).shape == (0, config.EMBEDDING_DIMENSION)
    vector = embedder.embed([""])[0]  # no features -> zero vector, but no crash
    assert vector.shape == (config.EMBEDDING_DIMENSION,)


def test_single_string_instead_of_list_is_rejected(embedder: HashingEmbedder):
    with pytest.raises(TypeError):
        embedder.embed("not a list")  # type: ignore[arg-type]


def test_dimension_matches_config_for_both_backends(embedder: HashingEmbedder):
    """Both chunking strategies share one embedder, hence one compatible dimension."""
    assert embedder.dimension == config.EMBEDDING_DIMENSION == 384


def test_get_embedder_caches_instance_and_validates_backend(monkeypatch):
    reset_embedder_cache()
    first = get_embedder("hashing")
    assert get_embedder("hashing") is first
    with pytest.raises(ValueError):
        get_embedder("does-not-exist")
    reset_embedder_cache()


def test_auto_falls_back_when_model_unavailable(monkeypatch):
    reset_embedder_cache()

    def boom() -> None:
        raise EmbeddingError("simulated: no internet")

    monkeypatch.setattr(embeddings, "_load_sentence_transformer", boom)
    monkeypatch.setattr(config, "ALLOW_FALLBACK_EMBEDDER", True)
    assert isinstance(get_embedder("auto"), HashingEmbedder)
    reset_embedder_cache()
    monkeypatch.setattr(config, "ALLOW_FALLBACK_EMBEDDER", False)
    with pytest.raises(EmbeddingError):
        get_embedder("auto")
    reset_embedder_cache()


def _real_model_or_none():
    try:
        return SentenceTransformerEmbedder(config.EMBEDDING_MODEL, local_only=True)
    except Exception:
        return None


def test_real_minilm_model_embeddings():
    """Runs only when all-MiniLM-L6-v2 is available locally; otherwise reported as SKIPPED."""
    model = _real_model_or_none()
    if model is None:
        pytest.skip("all-MiniLM-L6-v2 is not available offline on this machine (run python main.py once with internet)")
    vectors = model.embed(["Doctors use AI to detect disease.", "Physicians apply artificial intelligence to diagnose illness.", "Bananas are yellow."])
    assert vectors.shape == (3, config.EMBEDDING_DIMENSION)
    assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])
