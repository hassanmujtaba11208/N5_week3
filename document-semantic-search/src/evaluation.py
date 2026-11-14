"""Compare the two chunking strategies by running real searches.

For every query, both strategies are searched (top 5 by default) and each retrieved chunk
is recorded with its document, page, similarity, rank and the time the search took.
Nothing is hard-coded: the report is produced by running the implemented pipeline.

Relevance heuristic (transparent and automatic, NOT human judgement)
--------------------------------------------------------------------
Each evaluation query lists a few keywords (``config.EVAL_QUERIES``) that a genuinely
relevant passage is expected to mention. A retrieved chunk is counted as *relevant* when
it contains at least ``RELEVANT_MIN_KEYWORDS`` of them. "Precision@k" is the share of the
k results that are relevant. This is a rough proxy, so the reports always show the
retrieved text next to the numbers so that a person can judge for themselves.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

import config
from src.search import get_default_store, timed_search
from src.vector_store import VectorStore
from src.embeddings import Embedder
from src.utils import get_logger

logger = get_logger(__name__)

RELEVANT_MIN_KEYWORDS = 3
TIMING_REPEATS = 3

ROW_COLUMNS = [
    "query", "strategy", "rank", "document", "page", "similarity", "distance",
    "keyword_matches", "relevant", "chunk_chars", "execution_time_ms",
    "chunk_id", "retrieved_chunk",
]


@dataclass
class EvaluationResult:
    rows: pd.DataFrame
    summary: pd.DataFrame  # one row per (query, strategy)
    overall: pd.DataFrame  # one row per strategy
    embedding_backend: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    generated_at: str


def _keyword_matches(text: str, keywords: Sequence[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lowered)


def run_evaluation(
    queries: Optional[Sequence[Dict[str, Any]]] = None,
    top_k: int = config.EVAL_TOP_K,
    store: Optional[VectorStore] = None,
    embedder: Optional[Embedder] = None,
    strategies: Sequence[str] = config.STRATEGIES,
) -> EvaluationResult:
    """Run every query against every strategy and collect rows plus summaries."""
    query_specs = list(queries) if queries is not None else config.EVAL_QUERIES
    if not query_specs:
        raise ValueError("at least one evaluation query is required")
    vector_store = store or get_default_store()
    model = embedder or vector_store.embedder

    # Warm-up so one-off start-up cost (model, index load) does not distort the first timing.
    for strategy in strategies:
        timed_search(query_specs[0]["query"], top_k, strategy, store=vector_store, embedder=model)

    records: List[Dict[str, Any]] = []
    for spec in query_specs:
        query = spec["query"]
        keywords = list(spec.get("keywords", []))
        for strategy in strategies:
            timings: List[float] = []
            hits: List[Dict[str, Any]] = []
            for _ in range(TIMING_REPEATS):
                hits, elapsed = timed_search(query, top_k, strategy, store=vector_store, embedder=model)
                timings.append(elapsed)
            median_ms = round(statistics.median(timings), 2)
            for hit in hits:
                matches = _keyword_matches(hit["text"], keywords) if keywords else 0
                records.append(
                    {
                        "query": query,
                        "strategy": strategy,
                        "rank": hit["rank"],
                        "document": hit["document_name"],
                        "page": hit["page_number"],
                        "similarity": hit["similarity"],
                        "distance": hit["distance"],
                        "keyword_matches": matches,
                        "relevant": bool(keywords) and matches >= min(RELEVANT_MIN_KEYWORDS, len(keywords)),
                        "chunk_chars": len(hit["text"]),
                        "execution_time_ms": median_ms,
                        "chunk_id": hit["chunk_id"],
                        "retrieved_chunk": hit["text"],
                    }
                )
    rows = pd.DataFrame(records, columns=ROW_COLUMNS)
    summary = _summarise(rows, ["query", "strategy"])
    overall = _summarise(rows, ["strategy"])

    meta_collection = vector_store.get_collection(strategies[0])
    backend = str((meta_collection.metadata or {}).get("embedding_backend", model.name))
    return EvaluationResult(
        rows=rows, summary=summary, overall=overall, embedding_backend=backend, top_k=top_k,
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def _summarise(rows: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    grouped = rows.groupby(by, sort=False)
    summary = grouped.agg(
        results=("rank", "count"),
        mean_similarity=("similarity", "mean"),
        top1_similarity=("similarity", "max"),
        precision_at_k=("relevant", "mean"),
        mean_keyword_matches=("keyword_matches", "mean"),
        distinct_documents=("document", "nunique"),
        avg_chunk_chars=("chunk_chars", "mean"),
        avg_time_ms=("execution_time_ms", "mean"),
    ).reset_index()
    for column in ("mean_similarity", "top1_similarity", "precision_at_k", "mean_keyword_matches", "avg_chunk_chars", "avg_time_ms"):
        summary[column] = summary[column].round(3 if column != "avg_chunk_chars" else 0)
    return summary


# --------------------------------------------------------------------------- reports
def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _excerpt(text: str, limit: int = 320) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _md_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    lines = [header, divider]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_md_escape(str(v)) for v in row) + " |")
    return "\n".join(lines)


def _compare(summary: pd.DataFrame, query: str) -> str:
    """Programmatic, data-driven comparison sentence (no hand-written conclusions)."""
    part = summary[summary["query"] == query].set_index("strategy")
    if not {config.STRATEGY_FIXED, config.STRATEGY_RECURSIVE} <= set(part.index):
        return ""
    fixed, recursive = part.loc[config.STRATEGY_FIXED], part.loc[config.STRATEGY_RECURSIVE]

    def verdict(metric: str, label: str) -> str:
        a, b = float(fixed[metric]), float(recursive[metric])
        if abs(a - b) < 1e-9:
            return f"{label}: tie ({a})"
        winner = config.STRATEGY_FIXED if a > b else config.STRATEGY_RECURSIVE
        return f"{label}: **{winner}** higher (fixed {a} vs recursive {b})"

    return "; ".join([
        verdict("mean_similarity", "mean similarity"),
        verdict("precision_at_k", f"precision@{int(fixed['results'])}"),
        verdict("mean_keyword_matches", "keyword matches"),
    ])


def build_markdown_report(result: EvaluationResult) -> str:
    out: List[str] = []
    out.append("# Chunking Strategy Evaluation Report\n")
    out.append(f"*Generated automatically on {result.generated_at} by running the implemented pipeline.*\n")
    out.append("## Setup\n")
    out.append(f"- Embedding backend used for this run: **`{result.embedding_backend}`**")
    if result.embedding_backend.startswith("hashing"):
        out.append(
            "- ⚠️ **This run used the offline hashing fallback embedder, not `all-MiniLM-L6-v2`.** The fallback matches "
            "words rather than meaning, so similarity scores and rankings are NOT representative of the real model. "
            "Re-run `python main.py --reset` on a machine with internet access to regenerate this report with MiniLM."
        )
    out.append(f"- Chunk size: {result.chunk_size} characters, overlap: {result.chunk_overlap} characters")
    out.append(f"- Results per strategy per query (top-k): {result.top_k}")
    out.append("- Scores: **cosine similarity** = 1 − ChromaDB cosine distance (range −1..1, higher is better; not a percentage)")
    out.append("- Execution time: median of "
               f"{TIMING_REPEATS} runs, in milliseconds, covering query embedding + vector search")
    out.append(f"- *Relevant* = chunk mentions at least {RELEVANT_MIN_KEYWORDS} of the query's expected keywords "
               "(automatic heuristic; read the retrieved text to judge for yourself)\n")

    out.append("## Overall summary (all queries)\n")
    out.append(_md_table(result.overall) + "\n")

    for query in result.rows["query"].drop_duplicates():
        out.append(f"## Query: {query}\n")
        qsum = result.summary[result.summary["query"] == query].drop(columns=["query"])
        out.append(_md_table(qsum) + "\n")
        comparison = _compare(result.summary, query)
        if comparison:
            out.append(f"**Automatic comparison:** {comparison}\n")
        for strategy in config.STRATEGIES:
            part = result.rows[(result.rows["query"] == query) & (result.rows["strategy"] == strategy)]
            if part.empty:
                continue
            out.append(f"### {strategy} chunking: top {len(part)} results\n")
            table = pd.DataFrame({
                "rank": part["rank"], "document": part["document"], "page": part["page"],
                "similarity": part["similarity"], "kw": part["keyword_matches"],
                "time_ms": part["execution_time_ms"],
                "retrieved chunk (excerpt)": part["retrieved_chunk"].map(_excerpt),
            })
            out.append(_md_table(table) + "\n")
    out.append("## Notes and limitations\n")
    out.append("- The CSV file `chunking_comparison.csv` contains the full text of every retrieved chunk.")
    out.append("- Keyword-based relevance is only a proxy; with three queries and three documents this is a small experiment, "
               "not a benchmark.")
    out.append("- Small differences in timing are within normal run-to-run noise on a laptop.\n")
    return "\n".join(out)


def write_reports(result: EvaluationResult, csv_path: Path | str | None = None,
                  md_path: Path | str | None = None) -> Dict[str, Path]:
    """Write the CSV and Markdown reports (defaults: reports/ folder) and return their paths."""
    csv_file = Path(csv_path) if csv_path is not None else config.CSV_REPORT_PATH
    md_file = Path(md_path) if md_path is not None else config.MD_REPORT_PATH
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    md_file.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = result.rows.assign(embedding_backend=result.embedding_backend)
    csv_rows["retrieved_chunk"] = csv_rows["retrieved_chunk"].map(lambda t: " ".join(str(t).split()))  # one line per row
    csv_rows.to_csv(csv_file, index=False, encoding="utf-8-sig")
    md_file.write_text(build_markdown_report(result), encoding="utf-8")
    logger.info("Wrote %s and %s", csv_file, md_file)
    return {"csv": csv_file, "markdown": md_file}
