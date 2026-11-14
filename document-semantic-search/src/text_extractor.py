"""Extract text from PDF files, page by page, using PyMuPDF."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pymupdf

from src.utils import clean_text, get_logger

logger = get_logger(__name__)


class ExtractionError(Exception):
    """Raised when a PDF cannot be opened or read at all."""


@dataclass
class PageText:
    """Text of a single PDF page (page_number is 1-based)."""

    page_number: int
    text: str


@dataclass
class ExtractionResult:
    """Everything we learned about one PDF."""

    pages: List[PageText] = field(default_factory=list)
    total_pages: int = 0
    empty_pages: int = 0
    title: str = ""
    author: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def has_text(self) -> bool:
        return any(page.text.strip() for page in self.pages)


def _page_paragraphs(page: "pymupdf.Page") -> List[str]:
    """Return the page's paragraphs (PyMuPDF text blocks) in reading order.

    PyMuPDF "blocks" roughly correspond to paragraphs. Line breaks *inside* a block are
    just visual wrapping, so they are replaced by spaces. Joining blocks with a blank
    line gives the recursive chunker real paragraph boundaries to respect.
    """
    blocks = page.get_text("blocks", sort=True)
    paragraphs: List[str] = []
    for block in blocks:
        # block = (x0, y0, x1, y1, text, block_no, block_type); type 0 is text
        if len(block) >= 7 and block[6] != 0:
            continue
        raw = str(block[4])
        raw = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)  # re-join words hyphenated at line end
        raw = re.sub(r"\s*\n\s*", " ", raw).strip()
        if raw:
            paragraphs.append(clean_text(raw))
    return [p for p in paragraphs if p]


def _signature(paragraph: str) -> str:
    """Normalise a paragraph so 'Page 3' and 'Page 4' look identical (digits -> #)."""
    return re.sub(r"\d+", "#", " ".join(paragraph.lower().split()))


def _repeated_signatures(per_page: List[List[str]]) -> set:
    """Find short blocks that repeat on most pages: running headers, footers, page numbers."""
    pages_with_text = [p for p in per_page if p]
    if len(pages_with_text) < 3:
        return set()  # too few pages to tell a footer from real content
    seen: dict = {}
    for paragraphs in pages_with_text:
        for signature in {_signature(p) for p in paragraphs if len(p) <= 200}:
            seen[signature] = seen.get(signature, 0) + 1
    threshold = max(2, math.ceil(0.6 * len(pages_with_text)))
    return {sig for sig, count in seen.items() if count >= threshold}


def extract_pdf_text(pdf_path: Path | str) -> ExtractionResult:
    """Extract text from every page of a PDF.

    Raises:
        FileNotFoundError: the path does not exist.
        ExtractionError: the file is not a readable PDF (corrupted, encrypted, empty file).
    Pages without text (e.g. scanned images) are kept out of ``pages`` and counted in
    ``empty_pages``; a warning is added so callers can report it.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.stat().st_size == 0:
        raise ExtractionError(f"PDF file is empty (0 bytes): {path.name}")

    try:
        document = pymupdf.open(path)
    except Exception as exc:  # pymupdf raises several exception types for bad files
        raise ExtractionError(f"Cannot open '{path.name}' as a PDF: {exc}") from exc

    result = ExtractionResult()
    try:
        if document.needs_pass:
            raise ExtractionError(f"PDF is password protected: {path.name}")
        result.total_pages = document.page_count
        meta = document.metadata or {}
        result.title = (meta.get("title") or "").strip()
        result.author = (meta.get("author") or "").strip()
        per_page: List[List[str]] = []
        for index in range(document.page_count):
            try:
                per_page.append(_page_paragraphs(document[index]))
            except Exception as exc:
                result.warnings.append(f"page {index + 1}: could not be read ({exc})")
                logger.warning("%s page %d unreadable: %s", path.name, index + 1, exc)
                per_page.append([])
        repeated = _repeated_signatures(per_page)
        removed = 0
        for index, paragraphs in enumerate(per_page):
            kept = [p for p in paragraphs if _signature(p) not in repeated]
            removed += len(paragraphs) - len(kept)
            text = clean_text("\n\n".join(kept))
            if text:
                result.pages.append(PageText(page_number=index + 1, text=text))
            else:
                result.empty_pages += 1
                result.warnings.append(f"page {index + 1}: no extractable text (scanned image or blank)")
        if removed:
            logger.info("%s: removed %d repeated header/footer block(s)", path.name, removed)
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"Failed while reading '{path.name}': {exc}") from exc
    finally:
        document.close()

    if not result.has_text:
        result.warnings.append("document contains no extractable text")
    logger.info("Extracted %s: %d/%d pages with text", path.name, len(result.pages), result.total_pages)
    return result
