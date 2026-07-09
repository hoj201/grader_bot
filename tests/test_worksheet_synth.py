from pathlib import Path

import fitz
import numpy as np

from worksheet_synth import latexmk_worksheet, write_on_image

DEMO_TEX = Path(__file__).parent.parent / "demo.tex"
FONT_PATH = Path(__file__).parent.parent / "fonts" / "HomemadeApple-Regular.ttf"


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


def test_write_on_image_draws_text_without_mutating_input():
    blank = np.full((100, 300, 3), 255, dtype=np.uint8)
    original = blank.copy()

    result = write_on_image(blank, "42", (10, 10), 24, str(FONT_PATH))

    assert blank.tolist() == original.tolist()
    assert result.shape == blank.shape
    assert not np.array_equal(result, blank)
