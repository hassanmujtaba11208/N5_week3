"""Streamlit user interface for the Document Semantic Search pipeline.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

import config
from src.embeddings import get_embedder
from src.evaluation import run_evaluation, write_reports
from src.pipeline import ingest
from src.search import InvalidQueryError, timed_search
from src.vector_store import CollectionNotFoundError, EmbeddingMismatchError, VectorStore

st.set_page_config(page_title="Document Semantic Search", page_icon="🔎", layout="wide")

MAX_UPLOAD_MB = 50
PAGES = ["Dashboard", "Document Ingestion", "Semantic Search", "Chunking Comparison", "Evaluation"]
STRATEGY_LABELS = {config.STRATEGY_FIXED: "Fixed-size chunking", config.STRATEGY_RECURSIVE: "Recursive character chunking"}

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      .result-meta {color:#5b6472; font-size:0.9rem; margin-bottom:0.4rem;}
      .score-pill {background:#e8f0fe; color:#1a4fb4; padding:2px 10px; border-radius:12px; font-weight:600;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading the embedding model (first run only)...")
def get_store() -> VectorStore:
    return VectorStore(config.CHROMA_DIR, get_embedder())


def embedding_banner(store: VectorStore) -> None:
    """Show index/embedder warnings: fallback embedder active, or index built with another model."""
    if store.embedder.backend == "hashing":
        st.warning(
            "**Offline fallback embedder is active.** The real `all-MiniLM-L6-v2` model could not be loaded, so a simple "
            "word-matching embedder is used instead. Search still works, but results are lexical, not semantic. "
            "Connect to the internet once, then run `python main.py --reset` and restart the app to use the real model."
        )
    built_with = store.stats()["embedding_backend"]
    if built_with and built_with != store.embedder.name:
        st.error(
            f"The stored index was built with `{built_with}`, but the app is now using `{store.embedder.name}`. "
            "Vectors from different models cannot be compared, so search is disabled until the index is rebuilt."
        )
        if st.button("Rebuild the index with the current embedding model", key="rebuild_index"):
            try:
                with st.spinner("Rebuilding the index..."):
                    ingest(store, reset=True)
                st.rerun()
            except Exception as exc:
                st.error(f"Rebuild failed: {exc}")


def show_hits(hits: List[Dict[str, Any]], key_prefix: str = "") -> None:
    for hit in hits:
        header = f"#{hit['rank']} · {hit['document_name']} · page {hit['page_number']} · similarity {hit['similarity']:.3f}"
        with st.expander(header, expanded=hit["rank"] <= 3):
            st.markdown(
                f"<div class='result-meta'>Rank <b>{hit['rank']}</b> · Document <b>{hit['document_name']}</b> · "
                f"Page <b>{hit['page_number']}</b> · <span class='score-pill'>cosine similarity {hit['similarity']:.4f}</span> "
                f"· ChromaDB distance {hit['distance']:.4f}</div>",
                unsafe_allow_html=True,
            )
            st.write(hit["text"])
            st.caption(f"chunk id: {hit['chunk_id']}")


def run_search(store: VectorStore, query: str, top_k: int, strategy: str):
    """Search and turn expected failures into friendly messages. Returns (hits, ms) or None."""
    try:
        return timed_search(query, top_k, strategy, store=store)
    except InvalidQueryError as exc:
        st.error(f"Invalid search: {exc}")
    except CollectionNotFoundError as exc:
        st.info(str(exc))
    except EmbeddingMismatchError as exc:
        st.error(str(exc))
    except Exception as exc:  # last-resort guard so the app never shows a raw traceback
        st.error(f"Search failed: {exc}")
    return None


# --------------------------------------------------------------------------- pages
def page_dashboard(store: VectorStore) -> None:
    st.title("🔎 Document Ingestion, Chunking & Semantic Search")
    st.write(
        "Ingest PDF documents, split them with two different chunking strategies, embed the chunks with "
        "**all-MiniLM-L6-v2**, store them in a persistent **ChromaDB** database and search them by meaning. "
        "Use the sidebar to ingest documents, search, and compare the strategies."
    )
    embedding_banner(store)
    stats = store.stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ingested documents", stats["documents"])
    c2.metric("Extracted pages", stats["pages_with_text"])
    c3.metric("Generated chunks", stats["total_chunks"])
    c4.metric("Stored embeddings", stats["embeddings"])
    d1, d2 = st.columns(2)
    d1.metric("Fixed-size chunks", stats["chunks"][config.STRATEGY_FIXED])
    d2.metric("Recursive chunks", stats["chunks"][config.STRATEGY_RECURSIVE])
    st.caption(
        f"Embedding backend: `{stats['embedding_backend'] or store.embedder.name}` · "
        f"database folder: `{stats['persist_dir']}`"
    )
    if stats["total_chunks"] == 0:
        st.info("The database is empty. Open **Document Ingestion** and ingest the sample documents to get started.")
    else:
        st.subheader("Indexed documents")
        st.write(", ".join(f"`{d}`" for d in stats["document_ids"]))


def page_ingestion(store: VectorStore) -> None:
    st.header("Document Ingestion")
    embedding_banner(store)
    uploads = st.file_uploader("Upload one or more PDF documents", type=["pdf"], accept_multiple_files=True)
    if uploads:
        st.write("**Selected files:**")
        for upload in uploads:
            st.write(f"- {upload.name} ({upload.size / 1024:.1f} KB)")

    col_a, col_b = st.columns(2)
    ingest_uploads = col_a.button("Ingest uploaded documents", type="primary", disabled=not uploads)
    ingest_folder = col_b.button(f"Re-ingest all PDFs in {config.DOCUMENTS_DIR.name}/ (sample + uploaded)")
    rebuild = st.checkbox("Rebuild from scratch (delete the existing index first)")

    if not (ingest_uploads or ingest_folder):
        st.caption("Ingesting the same document again replaces its chunks, so duplicates are never created.")
        return

    paths: Optional[List[Path]] = None
    if ingest_uploads and uploads:
        config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        paths = []
        for upload in uploads:
            name = Path(upload.name).name  # strip any path components
            data = upload.getvalue()
            if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"{name}: larger than {MAX_UPLOAD_MB} MB, skipped.")
                continue
            if not data.startswith(b"%PDF"):
                st.error(f"{name}: does not look like a PDF file, skipped.")
                continue
            target = config.DOCUMENTS_DIR / name
            target.write_bytes(data)
            paths.append(target)
        if not paths:
            st.error("No valid PDF files to ingest.")
            return

    bar = st.progress(0.0, text="Starting...")
    try:
        report, failures = ingest(
            store, paths=paths, reset=rebuild,
            progress=lambda fraction, label: bar.progress(min(fraction, 1.0), text=f"Processing {label}"),
        )
    except (ValueError, FileNotFoundError) as exc:
        bar.empty()
        st.error(f"Ingestion failed: {exc}")
        return
    except Exception as exc:
        bar.empty()
        st.error(f"Unexpected error during ingestion: {exc}")
        return
    bar.progress(1.0, text="Done")
    st.success(
        f"Ingested {len(report.documents)} document(s): "
        + ", ".join(f"{s}: {r.chunks} chunks" for s, r in report.strategies.items())
        + f" (embedding backend: {report.embedding_backend})"
    )
    for failure in failures:
        st.warning(f"Skipped {Path(failure.path).name}: {failure.reason}")


def page_search(store: VectorStore) -> None:
    st.header("Semantic Search")
    embedding_banner(store)
    query = st.text_input("Ask a question in natural language", placeholder="e.g. What are the benefits of AI in healthcare?")
    c1, c2 = st.columns(2)
    top_k = c1.slider("Top-K results", 1, 20, config.DEFAULT_TOP_K)
    strategy = c2.radio("Chunking strategy", list(STRATEGY_LABELS), format_func=STRATEGY_LABELS.get, index=1, horizontal=True)
    if st.button("Search", type="primary"):
        outcome = run_search(store, query, top_k, strategy)
        if outcome:
            hits, ms = outcome
            st.caption(f"{len(hits)} result(s) in {ms:.1f} ms · similarity = cosine similarity (1 − ChromaDB distance), higher is better")
            show_hits(hits)


def page_comparison(store: VectorStore) -> None:
    st.header("Chunking Comparison")
    embedding_banner(store)
    query = st.text_input("Query to compare", value=config.EVAL_QUERIES[0]["query"], key="cmp_query")
    top_k = st.slider("Top-K results per strategy", 1, 10, config.EVAL_TOP_K, key="cmp_k")
    if not st.button("Compare both strategies", type="primary"):
        return
    outcomes = {s: run_search(store, query, top_k, s) for s in config.STRATEGIES}
    if not all(outcomes.values()):
        return
    left, right = st.columns(2)
    for column, strategy in zip((left, right), config.STRATEGIES):
        hits, ms = outcomes[strategy]  # type: ignore[misc]
        with column:
            st.subheader(STRATEGY_LABELS[strategy])
            st.metric("Retrieval time", f"{ms:.1f} ms")
            show_hits(hits)
    rows = []
    for strategy in config.STRATEGIES:
        hits, ms = outcomes[strategy]  # type: ignore[misc]
        rows.append({
            "strategy": strategy,
            "mean similarity": round(sum(h["similarity"] for h in hits) / len(hits), 4),
            "top-1 similarity": hits[0]["similarity"],
            "avg chunk length (chars)": round(sum(len(h["text"]) for h in hits) / len(hits)),
            "distinct documents": len({h["document_name"] for h in hits}),
            "time (ms)": round(ms, 2),
        })
    st.subheader("Comparison metrics")
    st.dataframe(pd.DataFrame(rows).set_index("strategy"))


def page_evaluation(store: VectorStore) -> None:
    st.header("Evaluation")
    embedding_banner(store)
    st.write("Runs the predefined queries against both strategies (top 5 each) and generates real CSV and Markdown reports.")
    for item in config.EVAL_QUERIES:
        st.markdown(f"- {item['query']}")
    if st.button("Run evaluation", type="primary"):
        try:
            with st.spinner("Running searches..."):
                result = run_evaluation(store=store)
                write_reports(result)
            st.session_state["eval_result"] = result
            st.success("Evaluation finished and reports were written to the reports/ folder.")
        except CollectionNotFoundError as exc:
            st.info(str(exc))
        except EmbeddingMismatchError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

    result = st.session_state.get("eval_result")
    if result is not None:
        st.subheader("Overall summary")
        st.dataframe(result.overall.set_index("strategy"))
        st.subheader("Per-query summary")
        st.dataframe(result.summary)
        st.subheader("Comparison table (all retrieved chunks)")
        st.dataframe(result.rows)

    if config.CSV_REPORT_PATH.is_file() and config.MD_REPORT_PATH.is_file():
        d1, d2 = st.columns(2)
        d1.download_button("Download CSV report", config.CSV_REPORT_PATH.read_bytes(), file_name="chunking_comparison.csv", mime="text/csv")
        d2.download_button("Download Markdown report", config.MD_REPORT_PATH.read_bytes(), file_name="evaluation_report.md", mime="text/markdown")


# --------------------------------------------------------------------------- main
def main() -> None:
    try:
        store = get_store()
    except Exception as exc:
        st.error(f"Could not start the pipeline: {exc}")
        st.stop()
        return
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")
    st.sidebar.caption(f"Embedding: {store.embedder.name}")
    {
        "Dashboard": page_dashboard,
        "Document Ingestion": page_ingestion,
        "Semantic Search": page_search,
        "Chunking Comparison": page_comparison,
        "Evaluation": page_evaluation,
    }[page](store)


main()
