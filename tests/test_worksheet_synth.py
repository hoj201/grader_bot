from pathlib import Path

import fitz
import numpy as np
import pytest

from graderbot.core import extract_answer_boxes
from graderbot.worksheet_synth import (
    add_image_noise,
    fill_worksheet,
    latexmk_worksheet,
    perspective_skew_image,
    write_on_image,
)

DEMO_TEX = Path(__file__).parent.parent / "tex" / "demo.tex"
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


def test_latexmk_worksheet_finds_repo_root_style_files_from_subdirectory(tmp_path):
    """gbworksheet.sty/questions.sty live at the repo root, not in the TeX
    distribution. app.py writes generated .tex files into a subdirectory
    (generated/<uuid>/), so compilation must still find them via TEXINPUTS
    rather than relying on cwd."""
    tex_copy = tmp_path / "demo.tex"
    tex_copy.write_text(DEMO_TEX.read_text())

    pdf_path = latexmk_worksheet(str(tex_copy), cv_mode=False)

    assert Path(pdf_path).is_file()


def test_write_on_image_draws_text_without_mutating_input():
    blank = np.full((100, 300, 3), 255, dtype=np.uint8)
    original = blank.copy()

    result = write_on_image(blank, "42", (10, 10), 24, str(FONT_PATH))

    assert blank.tolist() == original.tolist()
    assert result.shape == blank.shape
    assert not np.array_equal(result, blank)


def test_perspective_skew_image_preserves_shape_without_mutating_input():
    image = np.full((100, 300, 3), 255, dtype=np.uint8)
    image[40:60, 100:200] = 0
    original = image.copy()

    result = perspective_skew_image(image, max_skew=0.05, rng=np.random.default_rng(0))

    assert image.tolist() == original.tolist()
    assert result.shape == image.shape


def test_perspective_skew_image_changes_image():
    image = np.full((100, 300, 3), 255, dtype=np.uint8)
    image[40:60, 100:200] = 0

    result = perspective_skew_image(image, max_skew=0.05, rng=np.random.default_rng(0))

    assert not np.array_equal(result, image)


def test_perspective_skew_image_is_identity_when_max_skew_is_zero():
    image = np.full((100, 300, 3), 255, dtype=np.uint8)
    image[40:60, 100:200] = 0

    result = perspective_skew_image(image, max_skew=0.0, rng=np.random.default_rng(0))

    assert np.array_equal(result, image)


def test_add_image_noise_preserves_shape_without_mutating_input():
    image = np.full((100, 300, 3), 255, dtype=np.uint8)
    image[40:60, 100:200] = 0
    original = image.copy()

    result = add_image_noise(image, noise_level=0.05, rng=np.random.default_rng(0))

    assert image.tolist() == original.tolist()
    assert result.shape == image.shape


def test_add_image_noise_changes_image():
    image = np.full((100, 300, 3), 255, dtype=np.uint8)
    image[40:60, 100:200] = 0

    result = add_image_noise(image, noise_level=0.05, rng=np.random.default_rng(0))

    assert not np.array_equal(result, image)


def test_add_image_noise_is_identity_when_noise_level_is_zero():
    image = np.full((100, 300, 3), 255, dtype=np.uint8)
    image[40:60, 100:200] = 0

    result = add_image_noise(image, noise_level=0.0, rng=np.random.default_rng(0))

    assert np.array_equal(result, image)


def _box_region(image: np.ndarray, box) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = int(box.x_lower_left * width)
    x1 = int((box.x_lower_left + box.width) * width)
    y1 = int((1 - box.y_lower_left) * height)
    y0 = int(y1 - box.height * height)
    return image[y0:y1, x0:x1]


@pytest.fixture(scope="module")
def boxes():
    cv_worksheet = latexmk_worksheet(str(DEMO_TEX), cv_mode=True)
    return extract_answer_boxes(cv_worksheet)


def test_fill_worksheet_draws_plain_answer_into_its_box(boxes):
    filled = fill_worksheet(str(DEMO_TEX), {"add001": "12"})

    filled_region = _box_region(filled, boxes["add001"])
    assert not np.all(filled_region == 255)


def test_fill_worksheet_draws_fraction_into_its_box(boxes):
    filled = fill_worksheet(str(DEMO_TEX), {"sub001": r"\frac{3}{5}"})

    filled_region = _box_region(filled, boxes["sub001"])
    assert not np.all(filled_region == 255)


def test_fill_worksheet_draws_student_name_into_name_box(boxes):
    filled = fill_worksheet(str(DEMO_TEX), {}, student_name="Jane Doe")

    assert "name" in boxes
    filled_region = _box_region(filled, boxes["name"])
    assert not np.all(filled_region == 255)


def test_fill_worksheet_raises_on_unknown_question_id():
    try:
        fill_worksheet(str(DEMO_TEX), {"does_not_exist": "1"})
    except KeyError:
        return
    raise AssertionError("Expected KeyError for unknown question id")
