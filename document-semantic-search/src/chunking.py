"""Two genuinely different chunking strategies.

* **Fixed-size** (``fixed``): a sliding window of ``chunk_size`` characters that moves
  forward by ``chunk_size - chunk_overlap``. It ignores the text structure, so a chunk
  may start or end in the middle of a word or sentence. Simple, fast, predictable.

* **Recursive character** (``recursive``): tries to split on the *largest* natural
  boundary first (paragraph break), then line break, sentence ends, clauses and finally
  spaces. Pieces are merged back together up to ``chunk_size`` and the tail of each chunk
  is repeated at the start of the next one (overlap). Paragraphs and sentences stay
  intact whenever possible.

Both work page by page, so every chunk knows its document and page number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Sequence

import config
from src.document_loader import Document
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """One piece of text plus everything needed to store and cite it."""

    chunk_id: str
    text: str
    document_id: str
    document_name: str
    source_file: str
    page_number: int
    chunking_strategy: str
    chunk_index: int  # position of the chunk inside its document (0-based, across pages)
    document_title: str = ""

    @property
    def char_count(self) -> int:
        return len(self.text)

    def metadata(self) -> Dict[str, object]:
        """Flat metadata dict (ChromaDB only accepts str/int/float/bool values)."""
        data = asdict(self)
        data.pop("text")
        data["char_count"] = self.char_count
        return data


# --------------------------------------------------------------------------- validation
def _validate(chunk_size: int, chunk_overlap: int) -> None:
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be a non-negative integer, got {chunk_overlap!r}")
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")


# --------------------------------------------------------------------------- Strategy A
def fixed_size_split(text: str, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP) -> List[str]:
    """Split ``text`` into fixed-length character windows with overlap."""
    _validate(chunk_size, chunk_overlap)
    text = text.strip()
    if not text:
        return []
    step = chunk_size - chunk_overlap
    pieces: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break  # the last window reached the end; do not emit a tail that is only overlap
        start += step
    return pieces


# --------------------------------------------------------------------------- Strategy B
def _split_keep_separator(text: str, separator: str) -> List[str]:
    """Split on ``separator`` but keep it attached to the end of the preceding piece."""
    parts = text.split(separator)
    pieces = [part + separator for part in parts[:-1]] + [parts[-1]]
    return [piece for piece in pieces if piece]


def _merge_pieces(pieces: Sequence[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    """Greedily join small pieces into chunks <= chunk_size, repeating a tail as overlap."""
    chunks: List[str] = []
    current: List[str] = []
    total = 0
    for piece in pieces:
        if current and total + len(piece) > chunk_size:
            joined = "".join(current).strip()
            if joined:
                chunks.append(joined)
            # keep a tail of whole pieces (<= overlap) so consecutive chunks share context
            while current and (total > chunk_overlap or total + len(piece) > chunk_size):
                total -= len(current.pop(0))
        current.append(piece)
        total += len(piece)
    joined = "".join(current).strip()
    if joined:
        chunks.append(joined)
    return chunks


def _recursive(text: str, separators: Sequence[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    separator = ""
    remaining: Sequence[str] = ()
    for index, candidate in enumerate(separators):
        if candidate == "" or candidate in text:
            separator = candidate
            remaining = separators[index + 1:]
            break

    if separator == "":  # no natural boundary left: fall back to a hard character split
        return fixed_size_split(text, chunk_size, chunk_overlap)

    pieces = _split_keep_separator(text, separator)
    result: List[str] = []
    small: List[str] = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            small.append(piece)
            continue
        if small:
            result.extend(_merge_pieces(small, chunk_size, chunk_overlap))
            small = []
        result.extend(_recursive(piece, remaining, chunk_size, chunk_overlap))
    if small:
        result.extend(_merge_pieces(small, chunk_size, chunk_overlap))
    return result


def recursive_split(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    separators: Sequence[str] = config.RECURSIVE_SEPARATORS,
) -> List[str]:
    """Split ``text`` using a hierarchy of separators, preserving paragraphs/sentences."""
    _validate(chunk_size, chunk_overlap)
    text = text.strip()
    if not text:
        return []
    return [chunk for chunk in _recursive(text, tuple(separators), chunk_size, chunk_overlap) if chunk.strip()]


# --------------------------------------------------------------------------- registry
SPLITTERS: Dict[str, Callable[..., List[str]]] = {
    config.STRATEGY_FIXED: fixed_size_split,
    config.STRATEGY_RECURSIVE: recursive_split,
}


def make_chunk_id(document_id: str, strategy: str, page_number: int, chunk_index: int) -> str:
    """Deterministic ID, e.g. ``ai_in_healthcare__recursive__p003__c0012``."""
    return f"{document_id}__{strategy}__p{page_number:03d}__c{chunk_index:04d}"


def chunk_document(
    document: Document,
    strategy: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Chunk]:
    """Chunk every page of ``document`` with the chosen strategy."""
    if strategy not in SPLITTERS:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Choose from {sorted(SPLITTERS)}")
    splitter = SPLITTERS[strategy]
    chunks: List[Chunk] = []
    index = 0
    for page in document.pages:
        for piece in splitter(page.text, chunk_size, chunk_overlap):
            if len(piece.strip()) < min(config.MIN_CHUNK_CHARS, chunk_size):
                continue
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document.document_id, strategy, page.page_number, index),
                    text=piece,
                    document_id=document.document_id,
                    document_name=document.document_name,
                    source_file=document.source_file,
                    page_number=page.page_number,
                    chunking_strategy=strategy,
                    chunk_index=index,
                    document_title=document.title,
                )
            )
            index += 1
    return chunks


def chunk_documents(
    documents: Iterable[Document],
    strategy: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Chunk]:
    """Chunk several documents with one strategy."""
    all_chunks: List[Chunk] = []
    for document in documents:
        all_chunks.extend(chunk_document(document, strategy, chunk_size, chunk_overlap))
    logger.info("Strategy '%s' produced %d chunk(s)", strategy, len(all_chunks))
    return all_chunks
