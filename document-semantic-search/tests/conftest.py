"""Shared test fixtures.

Tests build their own small PDFs with PyMuPDF and use the deterministic offline
``HashingEmbedder`` so they are fast and need no internet connection. One extra test
exercises the real sentence-transformers model and is skipped (and reported as skipped)
when that model is not available on the machine running the tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pymupdf
import pytest

from src.embeddings import HashingEmbedder
from src.vector_store import VectorStore

PAGE_TEXTS: Dict[str, List[str]] = {
    "health_doc": [
        "Artificial intelligence helps doctors detect disease earlier. Hospitals use imaging models to find tumours "
        "in scans. Clinicians receive alerts about patients whose condition may worsen. Treatment can be personalised "
        "for each patient using clinical data and genomic information.",
        "Remote monitoring lets patients with chronic illness stay at home. Wearable sensors stream heart rate data "
        "to the care team. Hospitals also forecast admissions to schedule staff and beds efficiently.",
        "Medical AI must be validated on local patients before use. Bias in clinical data can harm minority groups. "
        "Doctors remain responsible for every diagnosis and treatment decision.",
    ],
    "ml_doc": [
        "Machine learning models learn patterns from training data. Each training example contains features and a "
        "label. The model adjusts its parameters to reduce the loss on the training set using gradient descent.",
        "After training the model makes predictions on new inputs. Inference applies the learned parameters to unseen "
        "features. Overfitting happens when a model memorises the training data and fails on validation data.",
        "Evaluation uses precision recall and accuracy on a held out test set. Cross validation estimates how well "
        "predictions generalise to new data.",
    ],
    "security_doc": [
        "Data security protects information from unauthorised access. Encryption protects stored files and network "
        "traffic. Strong access control and multi factor authentication reduce the risk of a breach.",
        "Privacy law requires a lawful basis to process personal data. Organisations must minimise collection and "
        "delete data when it is no longer needed. Regulators can fine companies after a serious breach.",
        "Attackers use phishing and stolen passwords. Poisoning attacks corrupt training data. Incident response "
        "plans help teams contain a breach quickly and notify affected people.",
    ],
}


def make_pdf(path: Path, pages: List[str], title: str = "") -> Path:
    """Create a small PDF where each string becomes one page of text."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            rc = page.insert_textbox(pymupdf.Rect(50, 60, 545, 780), text, fontsize=11)
            assert rc >= 0, "test text did not fit on the page"
    if title:
        doc.set_metadata({"title": title})
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture()
def docs_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "documents"
    folder.mkdir()
    for name, pages in PAGE_TEXTS.items():
        make_pdf(folder / f"{name}.pdf", pages)
    return folder


@pytest.fixture()
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture()
def store(tmp_path: Path, embedder: HashingEmbedder):
    vector_store = VectorStore(tmp_path / "chroma", embedder)
    yield vector_store
    vector_store.close()


@pytest.fixture()
def indexed_store(store: VectorStore, docs_dir: Path):
    """A store already filled with the three tiny test documents, both strategies."""
    from src.pipeline import ingest

    ingest(store, directory=docs_dir, chunk_size=300, chunk_overlap=60)
    return store
