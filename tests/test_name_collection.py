"""Tests for the name-collection worksheet template (issue #41).

This worksheet gathers labelled handwriting samples: graderbot prints a target
name in the exemplar box at the top, and the student copies it into a grid of
blank boxes. Every grid box must be discoverable by `extract_answer_boxes` (so a
scan can be cropped into per-box samples). The exemplar box is likewise
discoverable in cv mode -- as `printedname` -- so ingest can crop it and OCR the
printed name as each sheet's label (issue #2); in blank (student-facing) mode it
is a plain box holding the name.
"""

import shutil
from pathlib import Path

import pytest

from graderbot.worksheet_boxes import extract_answer_boxes
from graderbot.worksheet_synth import latexmk_worksheet

TEMPLATE_TEX = Path(__file__).parent.parent / "tex" / "name_collection_template.tex"

NAME_GRID_ROWS = 9  # must match `name grid rows` in the template (one box each)
GRID_BOX_IDS = {f"name{i}" for i in range(1, NAME_GRID_ROWS + 1)}
PRINTED_NAME_BOX_ID = "printedname"
EXPECTED_BOX_IDS = GRID_BOX_IDS | {PRINTED_NAME_BOX_ID}


def _fill(worksheet_id: str | None = None, student_name: str = "Jane Doe") -> str:
    """Substitute the template placeholders. This worksheet has no questions, so
    it deliberately omits the %%QUESTIONS%% marker and is not filled via
    `fill_template`; the id is injected the same way `fill_template` would."""
    id_setup = (
        rf"\WorksheetSetup{{worksheet id={worksheet_id}}}" if worksheet_id else ""
    )
    return (
        TEMPLATE_TEX.read_text()
        .replace("%%WORKSHEET_ID%%", id_setup)
        .replace("%%STUDENT_NAME%%", student_name)
    )


def test_template_has_expected_placeholders():
    template = TEMPLATE_TEX.read_text()
    assert "%%STUDENT_NAME%%" in template
    assert "%%WORKSHEET_ID%%" in template


def test_cv_render_exposes_every_grid_box_and_printed_name(tmp_path):
    """Each blank grid box and the exemplar name box are red, id-labelled
    rectangles, so they all come back from extract_answer_boxes with the
    expected ids (the grid boxes plus `printedname`)."""
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk is not installed")

    tex_path = tmp_path / "name_collection.tex"
    tex_path.write_text(_fill())

    pdf_path = latexmk_worksheet(str(tex_path), cv_mode=True)
    boxes = extract_answer_boxes(pdf_path)

    assert set(boxes) == EXPECTED_BOX_IDS


def test_cv_exemplar_box_id_excludes_printed_name(tmp_path):
    """In cv mode the exemplar box drops the printed name and shows only the
    `printedname` id, so the name text never leaks into the extracted box id
    (extract_answer_boxes concatenates all text inside a red rectangle)."""
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk is not installed")

    tex_path = tmp_path / "name_collection.tex"
    tex_path.write_text(_fill(student_name="Alexander Hamilton"))

    pdf_path = latexmk_worksheet(str(tex_path), cv_mode=True)
    boxes = extract_answer_boxes(pdf_path)

    assert PRINTED_NAME_BOX_ID in boxes
    assert not any("Alexander" in box_id for box_id in boxes)


def test_blank_render_compiles(tmp_path):
    """The student-facing (blank) render compiles and has plain boxes, so
    extract_answer_boxes (which keys on red) finds none of them."""
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk is not installed")

    tex_path = tmp_path / "name_collection.tex"
    tex_path.write_text(_fill())

    pdf_path = latexmk_worksheet(str(tex_path), cv_mode=False)
    assert Path(pdf_path).exists()
    assert extract_answer_boxes(pdf_path) == {}
