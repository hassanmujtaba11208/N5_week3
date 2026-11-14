# Document Ingestion, Chunking & Semantic Search Pipeline

A small, complete project that reads PDF documents, cuts them into chunks in **two different ways**, turns the chunks
into vectors (embeddings), stores them in a **ChromaDB** database on your disk, and lets you **search by meaning**.
It also measures which chunking strategy retrieves better results and writes a CSV and a Markdown report.

> ## ⚠️ Read this first — two honest notes about this delivery
>
> **1. The three sample PDFs are original documents, not downloaded papers.**
> The environment that built this project had no access to public document websites, so it could not download real
> third-party PDFs. Instead, three original educational documents (4–5 pages each, roughly 1,400–2,000 words each) were written
> for this assignment and rendered to PDF. Their origin is stated in the footer of every page, in each PDF's metadata, and
> in [`data/documents/SOURCES.md`](data/documents/SOURCES.md). They are **not** presented as external publications.
> If your assignment needs third-party documents, copy any PDFs into `data/documents/` and run `python main.py --reset`.
>
> **2. The reports shipped in `reports/` were produced with the offline fallback embedder, not MiniLM.**
> The build environment also could not download `all-MiniLM-L6-v2` (the model host was blocked). The code uses MiniLM by
> default, but when the model cannot be loaded it falls back to a simple word-matching embedder so everything still runs.
> The shipped `reports/*` and `database/chroma_db/` were generated that way and say so clearly (`hashing:384d`).
> **On your laptop, run `python main.py --reset` once with internet access** — it downloads MiniLM, rebuilds the database
> and regenerates both reports with the real model. The MiniLM code path is implemented but could not be executed in the
> build environment; the test that exercises it is skipped when the model is missing and will run on your machine.

---

## 1. What the project does

1. **Ingests PDFs** from `data/documents/` (or uploaded in the app) and extracts the text of every page with PyMuPDF.
2. **Chunks** the text with two strategies: *fixed-size* and *recursive character*.
3. **Embeds** each chunk with `sentence-transformers` (`all-MiniLM-L6-v2`, 384 numbers per chunk).
4. **Stores** chunks, vectors and metadata in a persistent **ChromaDB** database (one collection per strategy).
5. **Searches**: `search(query, top_k, strategy)` returns the best chunks with document name, page and similarity.
6. **Evaluates**: runs three queries through both strategies, times them, and writes `reports/chunking_comparison.csv`
   and `reports/evaluation_report.md`.
7. Offers a **Streamlit app** with a dashboard, ingestion, search, side-by-side comparison and evaluation.

## 2. Why semantic search is useful

Keyword search only finds documents that contain the *same words* as your question. Semantic search compares *meaning*,
so "ways AI helps doctors" can find a paragraph about "clinicians using machine learning to detect disease" even though
few words match. This is the retrieval step behind many question-answering and RAG (retrieval-augmented generation)
systems. How you cut the text into chunks strongly affects what gets retrieved — which is exactly what this project compares.

## 3. Project architecture

```
document-semantic-search/
├── data/
│   ├── documents/            3 sample PDFs + SOURCES.md (attribution)
│   └── processed/            (reserved for processed output)
├── database/chroma_db/       persistent ChromaDB database (created by main.py)
├── src/
│   ├── document_loader.py    find PDFs, load them into Document objects, collect failures
│   ├── text_extractor.py     PyMuPDF page-by-page extraction, header/footer removal, error handling
│   ├── chunking.py           fixed-size + recursive character chunking, deterministic chunk IDs
│   ├── embeddings.py         MiniLM embedder (cached) + labelled offline fallback embedder
│   ├── vector_store.py       ChromaDB wrapper: insert, no-duplicates, persistence, stats
│   ├── search.py             search() with validation, ranking, distance → similarity
│   ├── evaluation.py         strategy comparison, metrics, CSV + Markdown reports
│   ├── pipeline.py           shared "ingest" workflow used by main.py and the app
│   └── utils.py              logging, slugs, hashing, distance → similarity conversion
├── reports/                  chunking_comparison.csv, evaluation_report.md
├── tests/                    pytest suite (document loader, chunking, embeddings, vector store, search, evaluation)
├── scripts/                  code that built the sample PDFs (sample_content.py, generate_sample_documents.py)
├── app.py                    Streamlit application
├── main.py                   command-line ingestion + evaluation
├── config.py                 all settings in one place
├── requirements.txt  pytest.ini  .gitignore  LICENSE  README.md
```

## 4. Technologies used

| Purpose | Library |
|---|---|
| PDF text extraction | **PyMuPDF** (`import pymupdf`) |
| Embeddings | **sentence-transformers**, model `all-MiniLM-L6-v2` |
| Vector database | **ChromaDB** (persistent, cosine space) |
| User interface | **Streamlit** |
| Results tables | **Pandas** |
| Tests | **Pytest** |

No paid APIs, no API keys, no cloud services. Everything runs locally. Python 3.11 or 3.12 is recommended.

## 5. How document ingestion works

1. `list_pdf_files` finds `*.pdf` files in `data/documents/` (a missing folder gives a clear error).
2. `extract_pdf_text` opens each PDF with PyMuPDF and reads it **page by page**, keeping the 1-based **page number**.
   Text blocks (paragraphs) are joined with blank lines so the recursive chunker can see paragraph boundaries.
   Blocks that repeat on most pages (running headers, footers, page numbers) are removed so they do not pollute chunks.
3. Bad input is handled without crashing: a 0-byte file, a corrupted/non-PDF file, a password-protected PDF, a PDF with
   only scanned images, or a blank page are each reported with a warning or a clear message. In a folder with several
   files, the good ones are still ingested and the bad ones are listed as failures.
4. Each document gets metadata: `document_name`, `document_id` (a safe slug of the file name), `source_file`, content
   hash, total pages, and PDF title.

## 6. How the two chunking strategies work

Both strategies run **page by page** on the same documents, use the same defaults (800 characters, 150 overlap — see
`config.py`), never emit empty chunks, and create deterministic IDs such as `ai_in_healthcare__recursive__p003__c0012`.

**Strategy A — Fixed-size.** A window of `chunk_size` characters slides forward by `chunk_size − overlap`. It ignores the
text structure, so chunks can begin or end in the middle of a word or sentence. It is simple, fast and predictable.

**Strategy B — Recursive character.** Tries the biggest natural boundary first: paragraph break `\n\n`, then line break,
then sentence ends (`. ? !`), then `;`, `,`, spaces, and finally single characters. Small pieces are merged back up to
`chunk_size`, and the tail of each chunk (up to `overlap` characters, in whole pieces) is repeated at the start of the next.
Paragraphs and sentences stay together whenever possible.

## 7. How embeddings are generated

`src/embeddings.py` loads `all-MiniLM-L6-v2` **once per process** and keeps it in memory. The first run downloads it into
`models_cache/`; later runs load it from disk first and only use the network if the files are missing, so it is not
downloaded repeatedly. Vectors have 384 dimensions and are L2-normalised. Both strategies use the same embedder, so their
vectors always have compatible dimensions; the database records the embedder name and dimension and refuses to mix vectors
from different models (it tells you to rebuild instead).

*Fallback:* if the model cannot be loaded (for example no internet on the very first run), `hashing:384d` — a deterministic
word/bigram-hashing embedder — is used and **clearly reported** in the logs, the database, the reports and the app. It matches
words, not meaning, so treat its results as a demo only. Set `ALLOW_FALLBACK_EMBEDDER=0` to make a missing model a hard error.

## 8. How ChromaDB stores information

The database folder `database/chroma_db/` holds **two collections**: `docsearch_fixed` and `docsearch_recursive`, so the strategies
are always separately identifiable. Each stored record has:

- **id** — the chunk ID (unique and stable)
- **embedding** — the 384-number vector
- **document** — the chunk text
- **metadata** — `document_name`, `document_id`, `page_number`, `chunk_id`, `chunking_strategy`, `chunk_index`,
  `source_file`, `char_count`, `document_title`, plus `content_hash` and the chunk settings used

Collections use **cosine** space. The database is written to disk and can be closed and reopened without losing data
(`main.py` proves this on every run, and a test checks it). Re-running ingestion is safe: a document's old chunks are deleted
and its new chunks are upserted by ID, so **no duplicates** are created.

## 9. How semantic search works

```python
from src.search import search
hits = search("What are the main benefits of AI in healthcare?", top_k=5, strategy="recursive")
for hit in hits:
    print(hit["rank"], hit["similarity"], hit["document_name"], hit["page_number"], hit["text"][:80])
```

1. The query is validated (non-empty string, `top_k` between 1 and 50, strategy `fixed` or `recursive`) — otherwise
   `InvalidQueryError`.
2. The query is embedded with the same model as the chunks.
3. ChromaDB returns the nearest chunks with their **cosine distance**.
4. Distance is converted to **similarity** and results are returned best first with `rank`, `similarity`, `distance`,
   `document_name`, `page_number`, `chunk_id`, `chunk_index`, `chunking_strategy`, `source_file` and `text`.

If the strategy has not been indexed yet, `CollectionNotFoundError` explains how to build it.

## 10. How the evaluation compares the strategies

`python main.py` (or the **Evaluation** page) runs the three queries in `config.EVAL_QUERIES` against both strategies and
retrieves the top 5 for each:

1. What are the main benefits of artificial intelligence in healthcare?
2. How does machine learning use training data to make predictions?
3. What are the main challenges of data security and privacy?

For every retrieved chunk it records query, strategy, rank, document, page, similarity, distance, retrieved text and the
**execution time** (median of 3 runs after a warm-up; covers query embedding + search). It then computes per-query and overall
metrics: mean and top-1 similarity, number of distinct documents, average chunk length, and a **keyword-based precision@5**
(a chunk counts as relevant if it mentions at least 3 of the keywords listed for that query in `config.py`).

The keyword measure is only a transparent, automatic proxy — the report also shows the retrieved text next to the numbers so
you can judge relevance yourself. Nothing in the reports is typed by hand: the "Automatic comparison" lines are computed from
the numbers. With only three queries and three documents this is a small experiment, not a benchmark.

## 11. Installation instructions

You need Python 3.11 or 3.12 and VS Code (any editor works). Open the project folder in VS Code, open a terminal
(**Terminal → New Terminal**, PowerShell), and follow the next section. The first run needs internet access once, to download
the embedding model (about 90 MB) and the Python packages.

## 12. Exact Windows PowerShell commands

Create virtual environment:

```powershell
python -m venv venv
```

Activate:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run ingestion and evaluation:

```powershell
python main.py
```

Launch the application:

```powershell
streamlit run app.py
```

Run tests:

```powershell
pytest -v
```

Useful extras: `python main.py --reset` (wipe and rebuild the database), `python main.py --skip-eval` (ingest only).

## 13. How to run the application

Run `streamlit run app.py`; your browser opens at <http://localhost:8501>. Pages (sidebar):

- **Dashboard** — title, number of documents, extracted pages, chunks (per strategy and total), stored embeddings.
- **Document Ingestion** — upload PDFs, see their names, press *Ingest* (progress bar + success/error messages), or re-ingest
  everything in `data/documents/`. Re-ingesting never duplicates records.
- **Semantic Search** — type a question, choose Top-K and chunking strategy, press *Search*; results show rank, document,
  page, similarity and the matching text.
- **Chunking Comparison** — one query, both strategies side by side, with retrieval times and comparison metrics.
- **Evaluation** — runs the three predefined queries, shows the comparison tables and offers **CSV** and **Markdown** downloads.

If the index was built with a different embedding model than the app is using, a red banner offers a one-click rebuild.

## 14. How to run the tests

```powershell
pytest -v
```

The suite covers: PDF extraction, fixed-size and recursive chunking, empty/corrupted documents, embedding generation, ChromaDB
insertion, duplicate prevention, closing and reopening the database, search, top-K limits, metadata preservation, invalid
input, the evaluation reports, and `main.py` end to end. Tests build their own tiny PDFs and use the offline embedder, so they
are fast and need no internet. One test uses the real MiniLM model and reports **SKIPPED** if the model is not on the machine.

## 15. How to interpret similarity scores

ChromaDB returns a **distance**, not a similarity. With cosine space:

```
distance   = 1 − cosine_similarity       (0 = same direction, 1 = unrelated, 2 = opposite;  lower is better)
similarity = 1 − distance                (−1 … 1;  higher is better)
```

This project reports both columns and labels them clearly. **The similarity is a cosine score, not a percentage and not a
probability** — 0.45 does not mean "45 % relevant". Use it to *rank* results and to compare them against each other. As a rough
guide for MiniLM, scores above about 0.5 usually indicate a strongly related passage, 0.3–0.5 a loosely related one, and below
about 0.2 mostly unrelated text — but the best judge is always reading the retrieved chunk. Scores from different embedding
models are not comparable (and the offline fallback embedder produces much lower scores).

## 16. Expected outputs

After `python main.py` you should see (counts are for the three bundled PDFs; chunk numbers are deterministic):

```
Persistence check passed: {'fixed': 52, 'recursive': 54} chunks found after reopening the database

=== Ingestion summary ===
documents ingested : 3 (13 pages with text)
chunks [fixed    ] : 52
chunks [recursive] : 54
stored embeddings  : 106  (backend: sentence-transformers:all-MiniLM-L6-v2)

=== Evaluation summary (mean over the 3 queries) ===
 strategy  results  mean_similarity  top1_similarity  precision_at_k  ...
    fixed       15            ...
recursive       15            ...
CSV report      : reports\chunking_comparison.csv
Markdown report : reports\evaluation_report.md
```

Similarity and timing numbers depend on the embedding backend and your machine, so they will differ from the shipped
(fallback-embedder) reports. The CSV has one row per query × strategy × rank (30 rows) with the columns `query, strategy, rank,
document, page, similarity, distance, keyword_matches, relevant, chunk_chars, execution_time_ms, chunk_id, retrieved_chunk,
embedding_backend`. The Markdown report contains the setup, overall and per-query summary tables, and the top-5 results of
each strategy for each query.

## 17. Limitations and future improvements

- Only PDFs with a text layer are supported; scanned PDFs need OCR (not included). They are reported, not crashed on.
- Chunks are built per page, so a paragraph that continues on the next page is split at the page break.
- Chunk size is measured in characters, not tokens; MiniLM reads about 256 word-pieces, so the 800-character default fits.
- The relevance metric is keyword-based. A fairer evaluation would use human-labelled relevant passages (recall, MRR, nDCG).
- Only three queries and three short documents are evaluated; conclusions should not be generalised.
- Ideas: token-aware and semantic chunking, hybrid keyword + vector search, a re-ranker, OCR, more file types (DOCX, HTML),
  and passing retrieved chunks to a language model for answer generation (full RAG).
- Tested in the build environment on Linux with Python 3.13. The code uses `pathlib` and no OS-specific calls, but it was not
  run on Windows or Python 3.11/3.12 there — please run `pytest -v` on your machine to confirm.

## Troubleshooting

**`.\venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled`** — run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same terminal, then activate again.

**`pip install` fails or is very slow (torch is large)** — upgrade pip (`python -m pip install --upgrade pip`). For a smaller
CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `pip install -r requirements.txt`.

**ChromaDB fails to install on Windows (`Microsoft Visual C++ 14.0 or greater is required`)** — install the *Microsoft C++ Build
Tools*, or use Python 3.11/3.12 (prebuilt wheels exist) and upgrade pip.

**Model download fails / the log says "OFFLINE HASHING FALLBACK"** — check your internet connection, VPN or proxy and that
`huggingface.co` is reachable, then run `python main.py --reset`. The model is saved to `models_cache/` and reused afterwards. To
use a model you downloaded manually, put it in `models_cache/` or set `EMBEDDING_MODEL` to its folder path. In PowerShell:
`$env:EMBEDDING_MODEL = "C:\models\all-MiniLM-L6-v2"`.

**"Collection ... was built with X but the current embedder is Y"** — the database was made with a different model. Run
`python main.py --reset` (or press *Rebuild* in the app).

**ChromaDB errors or a corrupted database** — close the app and any running Python, then run `python main.py --reset`. If it
still fails, delete the `database\chroma_db` folder and run `python main.py` again (Windows cannot delete files that are in use).

**`ModuleNotFoundError: No module named 'src'` / `config`** — run commands from the project root folder (where `main.py` is).

**A PDF was skipped** — the log names the reason (corrupted, password-protected, or no extractable text). Scanned PDFs need OCR.

## Configuration

Everything is in `config.py`; most values can be overridden with environment variables, e.g.
`$env:CHUNK_SIZE = "600"; $env:CHUNK_OVERLAP = "100"; python main.py --reset`.
Main settings: `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `DEFAULT_TOP_K`, `CHROMA_DIR`, `DOCUMENTS_DIR`.

## License

MIT — see `LICENSE`.
