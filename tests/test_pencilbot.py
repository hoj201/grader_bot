import shutil
import subprocess
from pathlib import Path

import pytest

from pencilbot import extract_answer_boxes

DEMO_TEX = Path(__file__).parent.parent / "demo.tex"


@pytest.fixture(scope="module")
def demo_pdf(tmp_path_factory):
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk is not installed")

    build_dir = tmp_path_factory.mktemp("demo_build")
    subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            f"-output-directory={build_dir}",
            str(DEMO_TEX),
        ],
        cwd=DEMO_TEX.parent,
        check=True,
        capture_output=True,
    )
    return build_dir / "demo.pdf"


def test_extract_answer_boxes_finds_all_ids(demo_pdf):
    boxes = extract_answer_boxes(str(demo_pdf))
    assert set(boxes) == {"add001", "sub001"}


def test_extract_answer_boxes_coordinates_are_relative(demo_pdf):
    boxes = extract_answer_boxes(str(demo_pdf))
    for box in boxes.values():
        assert 0 <= box.x_lower_left <= 1
        assert 0 <= box.y_lower_left <= 1
        assert 0 < box.width <= 1
        assert 0 < box.height <= 1


def test_extract_answer_boxes_sub001_is_below_add001(demo_pdf):
    boxes = extract_answer_boxes(str(demo_pdf))
    assert boxes["sub001"].y_lower_left < boxes["add001"].y_lower_left
