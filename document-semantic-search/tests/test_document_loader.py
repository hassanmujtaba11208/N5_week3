"""Tests for PDF text extraction, loading, and graceful handling of bad files."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.document_loader import list_pdf_files, load_document, load_documents
from src.text_extractor import ExtractionError, extract_pdf_text
from tests.conftest import PAGE_TEXTS, make_pdf


def test_extracts_text_and_page_numbers(tmp_path: Path):
    pdf = make_pdf(tmp_path / "doc.pdf", PAGE_TEXTS["ml_doc"], title="ML test")
    result = extract_pdf_text(pdf)
    assert result.total_pages == 3
    assert [p.page_number for p in result.pages] == [1, 2, 3]
    assert "gradient descent" in result.pages[0].text
    assert "Overfitting" in result.pages[1].text
    assert result.title == "ML test"


def test_blank_page_is_reported_not_fatal(tmp_path: Path):
    pdf = make_pdf(tmp_path / "gap.pdf", ["Some real text on the first page of this file.", "", "Third page also has text."])
    result = extract_pdf_text(pdf)
    assert [p.page_number for p in result.pages] == [1, 3]
    assert result.empty_pages == 1
    assert any("page 2" in w for w in result.warnings)


def test_pdf_with_no_text_is_rejected_by_loader(tmp_path: Path):
    pdf = make_pdf(tmp_path / "blank.pdf", ["", ""])
    assert extract_pdf_text(pdf).has_text is False
    with pytest.raises(ExtractionError):
        load_document(pdf)


def test_corrupted_pdf_raises_extraction_error(tmp_path: Path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"this is definitely not a pdf file")
    with pytest.raises(ExtractionError):
        extract_pdf_text(bad)


def test_zero_byte_pdf_raises_extraction_error(tmp_path: Path):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(ExtractionError, match="empty"):
        extract_pdf_text(empty)


def test_missing_file_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        extract_pdf_text(tmp_path / "nope.pdf")


def test_repeated_footer_is_removed(tmp_path: Path):
    pages = [f"Unique body text number {i} about a different subject entirely, long enough to count." for i in range(4)]
    import pymupdf

    doc = pymupdf.open()
    for i, body in enumerate(pages, start=1):
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 60, 545, 700), body, fontsize=11)
        page.insert_text((50, 800), f"Confidential footer - page {i}", fontsize=8)
    target = tmp_path / "footer.pdf"
    doc.save(str(target))
    doc.close()
    result = extract_pdf_text(target)
    assert all("footer" not in p.text.lower() for p in result.pages)
    assert all("Unique body text" in p.text for p in result.pages)


def test_load_document_metadata(tmp_path: Path):
    pdf = make_pdf(tmp_path / "My Report (v2).pdf", PAGE_TEXTS["health_doc"])
    doc = load_document(pdf)
    assert doc.document_name == "My Report (v2).pdf"
    assert doc.document_id == "my_report_v2"
    assert doc.total_pages == 3 and doc.page_count_with_text == 3
    assert len(doc.content_hash) == 64
    assert doc.source_file.endswith("My Report (v2).pdf")


def test_load_documents_collects_failures(docs_dir: Path):
    (docs_dir / "broken.pdf").write_bytes(b"garbage")
    documents, failures = load_documents(directory=docs_dir)
    assert len(documents) == 3
    assert len(failures) == 1 and "broken.pdf" in failures[0].path


def test_list_pdf_files_sorted_and_validated(docs_dir: Path, tmp_path: Path):
    (docs_dir / "notes.txt").write_text("ignore me")
    names = [p.name for p in list_pdf_files(docs_dir)]
    assert names == sorted(names) and all(n.endswith(".pdf") for n in names) and len(names) == 3
    with pytest.raises(FileNotFoundError):
        list_pdf_files(tmp_path / "missing_folder")


def test_bundled_sample_documents_are_real_multi_page_pdfs():
    """The PDFs shipped in data/documents must be genuine multi-page documents."""
    import config

    pdfs = list_pdf_files(config.DOCUMENTS_DIR)
    assert len(pdfs) >= 3
    for pdf in pdfs:
        result = extract_pdf_text(pdf)
        assert result.total_pages >= 3, pdf.name
        words = sum(len(p.text.split()) for p in result.pages)
        assert words > 1000, f"{pdf.name} has only {words} words"
