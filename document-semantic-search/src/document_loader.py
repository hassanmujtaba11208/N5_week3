"""Discover PDF files and load them into :class:`Document` objects with metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import config
from src.text_extractor import ExtractionError, PageText, extract_pdf_text
from src.utils import file_sha256, get_logger, slugify

logger = get_logger(__name__)


@dataclass
class Document:
    """A successfully loaded document."""

    document_id: str
    document_name: str  # file name, e.g. "ai_in_healthcare.pdf"
    source_file: str  # path relative to the project root when possible
    content_hash: str
    total_pages: int
    pages: List[PageText] = field(default_factory=list)
    title: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def page_count_with_text(self) -> int:
        return len(self.pages)


@dataclass
class LoadFailure:
    """A document that could not be loaded (reported, never raised)."""

    path: str
    reason: str


def list_pdf_files(directory: Path | str | None = None) -> List[Path]:
    """Return PDF files in ``directory`` sorted by name.

    Raises FileNotFoundError if the directory does not exist.
    """
    folder = Path(directory) if directory is not None else config.DOCUMENTS_DIR
    if not folder.is_dir():
        raise FileNotFoundError(f"Documents folder not found: {folder}")
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_document(pdf_path: Path | str) -> Document:
    """Load one PDF. Raises FileNotFoundError / ExtractionError on failure."""
    path = Path(pdf_path)
    extraction = extract_pdf_text(path)
    if not extraction.has_text:
        raise ExtractionError(f"No extractable text in '{path.name}'")
    return Document(
        document_id=slugify(path.stem),
        document_name=path.name,
        source_file=_relative(path),
        content_hash=file_sha256(path),
        total_pages=extraction.total_pages,
        pages=extraction.pages,
        title=extraction.title,
        warnings=extraction.warnings,
    )


def load_documents(
    paths: Optional[Iterable[Path | str]] = None,
    directory: Path | str | None = None,
) -> Tuple[List[Document], List[LoadFailure]]:
    """Load many PDFs, collecting failures instead of stopping at the first bad file.

    If ``paths`` is None every PDF in ``directory`` (default: config.DOCUMENTS_DIR) is used.
    """
    candidates = [Path(p) for p in paths] if paths is not None else list_pdf_files(directory)
    documents: List[Document] = []
    failures: List[LoadFailure] = []
    seen_ids: set[str] = set()
    for path in candidates:
        try:
            doc = load_document(path)
        except (FileNotFoundError, ExtractionError) as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            failures.append(LoadFailure(path=str(path), reason=str(exc)))
            continue
        if doc.document_id in seen_ids:
            failures.append(LoadFailure(path=str(path), reason=f"duplicate document id '{doc.document_id}'"))
            continue
        seen_ids.add(doc.document_id)
        documents.append(doc)
    logger.info("Loaded %d document(s), %d failure(s)", len(documents), len(failures))
    return documents, failures
