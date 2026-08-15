import shutil
from pathlib import Path

import fitz
import pytest

from graderbot.imaging import render_pdf_page_image
from graderbot.registration import (
    _detect_marker_centers,
    read_worksheet_id,
    rectify_to_canonical,
)
from graderbot.worksheet_qr import generate_worksheet_id, render_qr_png
from graderbot.worksheet_synth import latexmk_worksheet
from graderbot.worksheetbot import fill_template

TEMPLATE_TEX = Path(__file__).parent.parent / "tex" / "worksheet_template.tex"

# Two questions separated by an explicit page break -- a deterministic way to
# force a two-page worksheet regardless of question-layout heuristics.
_TWO_PAGE_QUESTIONS = r"""
\Question{q1}{$1+1=$}

\newpage

\Question{q2}{$2+2=$}
"""


@pytest.fixture(scope="module")
def two_page_worksheet(tmp_path_factory):
    """A worksheet compiled from worksheet_template.tex (the same template
    graderbot fills for generated worksheets), spanning two pages.

    Regression fixture: \\WorksheetHeader used to invoke \\WorksheetMarkers
    inline, so the tikz overlay was only drawn on whichever page that inline
    call happened to fall on (page 1). Every later page shipped with no
    ArUco corner markers and no worksheet-id QR code, so scan_grader's
    per-page rectify_to_canonical/read_worksheet_id (see _grade_batch) failed
    on page 2+ of any real multi-page worksheet."""
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk is not installed")

    worksheet_id = generate_worksheet_id()
    tmp_path = tmp_path_factory.mktemp("two_page_worksheet")
    tex_source = fill_template(TEMPLATE_TEX, _TWO_PAGE_QUESTIONS, worksheet_id=worksheet_id)
    tex_path = tmp_path / "worksheet.tex"
    tex_path.write_text(tex_source)
    render_qr_png(worksheet_id, tmp_path / f"qr_{worksheet_id}.png")

    pdf_path = latexmk_worksheet(str(tex_path), cv_mode=False)
    with fitz.open(pdf_path) as doc:
        assert doc.page_count == 2, "fixture must actually span two pages to exercise the bug"

    return pdf_path, worksheet_id


def test_every_page_has_all_four_registration_markers(two_page_worksheet):
    pdf_path, _ = two_page_worksheet

    for page_index in range(2):
        image = render_pdf_page_image(pdf_path, page_index=page_index)
        centers = _detect_marker_centers(image, f"page {page_index}")
        assert sorted(centers.keys()) == [0, 1, 2, 3]


def test_every_page_rectifies_and_decodes_the_worksheet_id(two_page_worksheet):
    """Mirrors scan_grader._grade_batch, which rectifies to the canonical
    frame and decodes the worksheet id independently on every scanned page,
    then groups pages by the decoded id."""
    pdf_path, worksheet_id = two_page_worksheet

    for page_index in range(2):
        image = render_pdf_page_image(pdf_path, page_index=page_index)
        rectified = rectify_to_canonical(image)
        assert read_worksheet_id(rectified) == worksheet_id
