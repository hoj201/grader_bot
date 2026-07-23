"""Generate printable name-collection worksheets from a class roster (issue #45).

Given a list of student names, render one name-collection worksheet per name
(the issue #41 template) and merge them into a single PDF ready to print. Each
page prints one student's name at the top; the student copies it into the blank
grid below, producing labelled handwriting samples.

The sheets are rendered in blank (non-cv) mode, so the grid boxes are plain
black -- this is the student-facing print, not a scan to be cropped.
"""

import tempfile
from pathlib import Path
from typing import Iterable

import fitz

from graderbot.worksheet_synth import latexmk_worksheet
from graderbot.worksheetbot import escape_latex

_TEX_DIR = Path(__file__).resolve().parent.parent / "tex"
NAME_TEMPLATE_PATH = _TEX_DIR / "name_collection_template.tex"

STUDENT_NAME_MARKER = "%%STUDENT_NAME%%"
WORKSHEET_ID_MARKER = "%%WORKSHEET_ID%%"


def parse_roster(text: str) -> list[str]:
    """Split pasted roster text into a list of names, one per non-blank line."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def _fill_name_template(name: str) -> str:
    """Substitute the template placeholders for a single student name. The name
    is LaTeX-escaped so punctuation in a roster (e.g. ``O'Brien``, ``A & B``)
    cannot break compilation. No worksheet id is embedded -- these sheets are a
    print-only artefact, not stored worksheets."""
    template = NAME_TEMPLATE_PATH.read_text()
    return template.replace(WORKSHEET_ID_MARKER, "").replace(
        STUDENT_NAME_MARKER, escape_latex(name)
    )


def generate_name_worksheets(names: Iterable[str], out_path) -> Path:
    """Render one blank name-collection worksheet per name and merge them into a
    single PDF at ``out_path``, ready to print. Returns ``out_path``.

    Raises ``ValueError`` if no non-blank names are given.
    """
    names = [n for n in (name.strip() for name in names) if n]
    if not names:
        raise ValueError("No student names provided.")

    out_path = Path(out_path)
    merged = fitz.open()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for i, name in enumerate(names):
                tex_path = tmp_dir / f"name_{i}.tex"
                tex_path.write_text(_fill_name_template(name))
                pdf_path = latexmk_worksheet(str(tex_path), cv_mode=False)
                with fitz.open(pdf_path) as doc:
                    merged.insert_pdf(doc)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.save(str(out_path))
    finally:
        merged.close()
    return out_path
