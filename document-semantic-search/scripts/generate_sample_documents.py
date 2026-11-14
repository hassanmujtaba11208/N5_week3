"""Build the sample PDFs in data/documents/ from scripts/sample_content.py.

Usage (from the project root):
    python scripts/generate_sample_documents.py

The PDFs are original educational texts written for this project. Every PDF carries
an attribution line in its page footer and in its PDF metadata, and SOURCES.md repeats it.
You can replace these PDFs with any real PDFs: just copy them into data/documents/.
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_content import DOCUMENTS  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "documents"
AUTHOR = "Claude (AI assistant by Anthropic), written for this assignment"
CREATED = "2026-11-14"

ATTRIBUTION_PLAIN = (
    "Original educational document written for the Document Ingestion, Chunking and Semantic Search "
    "assignment by Claude (Anthropic) on " + CREATED + ". Not copied from any external publication."
)



FOOTER_NOTE = "Original educational text written for this assignment by Claude (Anthropic). Not an external publication."


def _footer(title: str):
    def draw(canvas, doc):  # type: ignore[no-untyped-def]
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, 1.2 * cm, FOOTER_NOTE)
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    return draw


def build_pdf(slug: str, spec: dict) -> Path:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", parent=styles["Title"], fontSize=21, leading=26, spaceAfter=6)
    sub_style = ParagraphStyle("S", parent=styles["Normal"], fontSize=12, textColor=colors.HexColor("#444444"), spaceAfter=14)
    h_style = ParagraphStyle("H", parent=styles["Heading2"], fontSize=14, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#1f3a5f"))
    body = ParagraphStyle("B", parent=styles["BodyText"], fontSize=11, leading=15.5, alignment=TA_JUSTIFY, spaceAfter=8)

    path = OUTPUT_DIR / f"{slug}.pdf"
    doc = SimpleDocTemplate(
        str(path), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2.2 * cm, bottomMargin=2.2 * cm,
        title=str(spec["title"]), author=AUTHOR, subject=ATTRIBUTION_PLAIN, creator="scripts/generate_sample_documents.py",
    )
    story = [Paragraph(str(spec["title"]), title_style), Paragraph(str(spec["subtitle"]), sub_style), Spacer(1, 4)]
    for heading, paragraphs in spec["sections"]:  # type: ignore[union-attr]
        story.append(Paragraph(heading, h_style))
        for p in paragraphs:
            story.append(Paragraph(p, body))
    footer = _footer(str(spec["title"]))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return path


def write_sources(paths: dict) -> None:
    lines = [
        "# Document sources and attribution", "",
        "All three sample PDFs are **original educational documents written for this project** "
        f"by Claude (an AI assistant made by Anthropic) on {CREATED}.",
        "They are **not** copies of external publications. The environment in which this project "
        "was built could not download third-party PDFs (external document hosts were blocked), "
        "so original, clearly labelled material was authored instead. The same attribution "
        "appears in the footer of every page and in each PDF's metadata.", "",
        "| File | Title | Author / origin | Licence |", "|---|---|---|---|",
    ]
    for slug, path in paths.items():
        lines.append(f"| `{path.name}` | {DOCUMENTS[slug]['title']} | {AUTHOR} | MIT (same as project) |")
    lines += [
        "", "## Using your own documents", "",
        "Any PDF placed in this folder is picked up automatically by `python main.py` and by the "
        "Streamlit *Document Ingestion* page. If your assignment requires third-party documents "
        "(for example public papers or reports), download them and copy them here, then re-run "
        "`python main.py --reset`. Remember to add their attribution to this file.", "",
        "Scripts used to build the samples: `scripts/sample_content.py`, "
        "`scripts/generate_sample_documents.py`.", "",
    ]
    (OUTPUT_DIR / "SOURCES.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    built = {slug: build_pdf(slug, spec) for slug, spec in DOCUMENTS.items()}
    write_sources(built)
    for slug, path in built.items():
        print(f"built {path.relative_to(PROJECT_ROOT)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
