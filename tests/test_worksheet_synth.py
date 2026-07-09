from pathlib import Path

import fitz

from worksheet_synth import latexmk_worksheet

DEMO_TEX = Path(__file__).parent.parent / "demo.tex"


def test_latexmk_worksheet_returns_pdf_for_cv_mode():
    pdf_path = latexmk_worksheet(str(DEMO_TEX), cv_mode=True)

    assert Path(pdf_path).is_file()
    assert Path(pdf_path).suffix == ".pdf"
    with fitz.open(pdf_path) as doc:
        assert doc.page_count > 0


def test_latexmk_worksheet_cv_and_blank_outputs_do_not_collide():
    cv_pdf_path = latexmk_worksheet(str(DEMO_TEX), cv_mode=True)
    blank_pdf_path = latexmk_worksheet(str(DEMO_TEX), cv_mode=False)

    assert cv_pdf_path != blank_pdf_path
    assert Path(cv_pdf_path).is_file()
    assert Path(blank_pdf_path).is_file()
