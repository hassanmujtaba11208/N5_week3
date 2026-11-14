"""Tests for the evaluation system, report generation and the CLI entry point."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import config
import main as main_module
from src.evaluation import ROW_COLUMNS, build_markdown_report, run_evaluation, write_reports
from src.pipeline import ingest
from src.vector_store import VectorStore

QUERIES = [
    {"query": "How do hospitals use AI to detect disease in patients?", "keywords": ["hospital", "patient", "disease"]},
    {"query": "How does a model learn from training data?", "keywords": ["training", "model", "data"]},
    {"query": "How do encryption and access control prevent a breach?", "keywords": ["encryption", "access", "breach"]},
]


def test_evaluation_rows_cover_every_query_strategy_and_rank(indexed_store: VectorStore):
    result = run_evaluation(QUERIES, top_k=5, store=indexed_store)
    assert list(result.rows.columns) == ROW_COLUMNS
    assert len(result.rows) == 3 * 2 * 5
    assert set(result.rows["strategy"]) == set(config.STRATEGIES)
    assert set(result.rows["query"]) == {q["query"] for q in QUERIES}
    for (_, _), group in result.rows.groupby(["query", "strategy"]):
        assert list(group["rank"]) == [1, 2, 3, 4, 5]
        assert group["similarity"].is_monotonic_decreasing
        assert (group["execution_time_ms"] > 0).all()
    assert result.rows["retrieved_chunk"].str.strip().ne("").all()
    assert result.embedding_backend.startswith("hashing")


def test_summary_tables(indexed_store: VectorStore):
    result = run_evaluation(QUERIES, top_k=3, store=indexed_store)
    assert len(result.summary) == 6 and len(result.overall) == 2
    assert result.summary["precision_at_k"].between(0, 1).all()
    # metrics must be computed from the rows, not invented
    manual = result.rows[result.rows["strategy"] == "fixed"]["similarity"].mean()
    assert result.overall.set_index("strategy").loc["fixed", "mean_similarity"] == pytest.approx(manual, abs=1e-3)


def test_reports_are_written_and_readable(indexed_store: VectorStore, tmp_path: Path):
    result = run_evaluation(QUERIES, top_k=5, store=indexed_store)
    paths = write_reports(result, tmp_path / "out" / "cmp.csv", tmp_path / "out" / "report.md")
    frame = pd.read_csv(paths["csv"])
    for column in ("query", "strategy", "document", "retrieved_chunk", "similarity", "rank", "execution_time_ms"):
        assert column in frame.columns
    assert len(frame) == 30
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "# Chunking Strategy Evaluation Report" in markdown
    for query in QUERIES:
        assert query["query"] in markdown
    assert "cosine similarity" in markdown and "hashing" in markdown  # backend + warning are disclosed
    assert "fixed chunking" in markdown and "recursive chunking" in markdown


def test_markdown_flags_fallback_backend(indexed_store: VectorStore):
    text = build_markdown_report(run_evaluation(QUERIES[:1], top_k=2, store=indexed_store))
    assert "offline hashing fallback" in text


def test_evaluation_requires_queries(indexed_store: VectorStore):
    with pytest.raises(ValueError):
        run_evaluation([], store=indexed_store)


def test_default_config_has_at_least_three_queries():
    assert len(config.EVAL_QUERIES) >= 3 and all(q["keywords"] for q in config.EVAL_QUERIES)


# ------------------------------------------------------------------------ main.py
@pytest.fixture()
def isolated_config(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "db")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "CSV_REPORT_PATH", tmp_path / "reports" / "c.csv")
    monkeypatch.setattr(config, "MD_REPORT_PATH", tmp_path / "reports" / "r.md")
    monkeypatch.setattr(config, "EMBEDDING_BACKEND", "hashing")
    return tmp_path


def test_main_runs_end_to_end_and_is_repeatable(isolated_config: Path, docs_dir: Path, monkeypatch):
    monkeypatch.setattr(config, "EVAL_QUERIES", QUERIES)
    assert main_module.main(["--documents-dir", str(docs_dir)]) == 0
    assert config.CSV_REPORT_PATH.is_file() and config.MD_REPORT_PATH.is_file()
    store = VectorStore(config.CHROMA_DIR, None)
    first = {s: store.count(s) for s in config.STRATEGIES}
    store.close()
    assert all(first.values())
    assert main_module.main(["--documents-dir", str(docs_dir), "--skip-eval"]) == 0  # repeat: no duplicates
    store = VectorStore(config.CHROMA_DIR, None)
    assert {s: store.count(s) for s in config.STRATEGIES} == first
    store.close()


def test_main_fails_cleanly_for_missing_folder_or_no_pdfs(isolated_config: Path, tmp_path: Path):
    assert main_module.main(["--documents-dir", str(tmp_path / "missing")]) == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main_module.main(["--documents-dir", str(empty)]) == 1


def test_ingest_reports_bad_files_but_continues(store: VectorStore, docs_dir: Path):
    (docs_dir / "corrupt.pdf").write_bytes(b"nope")
    report, failures = ingest(store, directory=docs_dir, chunk_size=300, chunk_overlap=60)
    assert len(report.documents) == 3 and len(failures) == 1
    only_bad = docs_dir.parent / "bad_only"
    only_bad.mkdir()
    (only_bad / "x.pdf").write_bytes(b"nope")
    with pytest.raises(ValueError, match="No documents could be ingested"):
        ingest(store, directory=only_bad)
