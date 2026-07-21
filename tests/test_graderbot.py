import os
import shutil
import subprocess
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytest

from graderbot import (
    Box,
    QuestionResult,
    _canonical_marker_centers,
    _detect_marker_centers,
    align_document_image,
    box_pixel_rect,
    extract_answer_boxes,
    extract_name,
    grade_hw,
    is_correct,
    load_image_rgb,
    read_box,
    rectify_to_canonical,
)
from worksheet_synth import fill_worksheet

DEMO_TEX = Path(__file__).parent.parent / "demo.tex"
DEMO_ANSWER_KEY_PDF = Path(__file__).parent.parent / "demo_answer_key.pdf"

_ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
_MARKER_IDS = (0, 1, 2, 3)  # top-left, top-right, bottom-left, bottom-right


def _render_pdf_page(pdf_path: Path, dpi: int):
    with fitz.open(str(pdf_path)) as doc:
        page = doc[0]
        zoom = dpi / 72
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
    return image


def _marker_relative_centers(image):
    detector = cv2.aruco.ArucoDetector(_ARUCO_DICTIONARY, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image)
    height, width = image.shape[:2]

    centers = {}
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        center = marker_corners[0].mean(axis=0)
        centers[int(marker_id)] = (center[0] / width, center[1] / height)

    assert set(centers) >= set(_MARKER_IDS)
    return centers


@pytest.fixture(scope="module")
def warped_photo_png(tmp_path_factory):
    reference_image = _render_pdf_page(DEMO_ANSWER_KEY_PDF, dpi=150)
    height, width = reference_image.shape[:2]

    src_corners = np.array(
        [[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32
    )
    inset = 0.02 * min(width, height)
    dst_corners = np.array(
        [
            [inset, 0.5 * inset],
            [width - 1.5 * inset, inset],
            [0.7 * inset, height - inset],
            [width - inset, height - 1.2 * inset],
        ],
        dtype=np.float32,
    )

    homography = cv2.getPerspectiveTransform(src_corners, dst_corners)
    warped_image = cv2.warpPerspective(
        reference_image,
        homography,
        (width, height),
        borderValue=(255, 255, 255),
    )

    output_dir = tmp_path_factory.mktemp("warped_photo")
    output_path = output_dir / "warped_photo.png"
    cv2.imwrite(str(output_path), warped_image)
    return output_path


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


@pytest.fixture(scope="module")
def boxes(demo_pdf):
    return extract_answer_boxes(str(demo_pdf))


def test_extract_answer_boxes_finds_all_ids(boxes):
    assert set(boxes) == {"name", "add001", "sub001", "frac001"}


def test_extract_answer_boxes_coordinates_are_relative(boxes):
    for box in boxes.values():
        assert 0 <= box.x_lower_left <= 1
        assert 0 <= box.y_lower_left <= 1
        assert 0 < box.width <= 1
        assert 0 < box.height <= 1


def test_extract_answer_boxes_sub001_is_below_add001(boxes):
    assert boxes["sub001"].y_lower_left < boxes["add001"].y_lower_left


def test_read_box_reads_handwritten_answers(boxes):
    if not os.environ.get("MATHPIX_APP_ID") or not os.environ.get("MATHPIX_APP_KEY"):
        pytest.skip("Mathpix credentials are not configured")

    answer_key_image = load_image_rgb(str(DEMO_ANSWER_KEY_PDF))

    assert read_box(answer_key_image, boxes["add001"]) == "12"
    assert read_box(answer_key_image, boxes["sub001"]) == "11"


def test_read_box_reads_a_handwritten_fraction(boxes):
    if not os.environ.get("MATHPIX_APP_ID") or not os.environ.get("MATHPIX_APP_KEY"):
        pytest.skip("Mathpix credentials are not configured")
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk is not installed")

    filled_image_bgr = fill_worksheet(str(DEMO_TEX), {"frac001": r"\frac{3}{4}"})
    filled_image = cv2.cvtColor(filled_image_bgr, cv2.COLOR_BGR2RGB)

    assert read_box(filled_image, boxes["frac001"]) == r"\frac{3}{4}"


def test_extract_name_matches_closest_roster_name(boxes):
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract is not installed")
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk is not installed")

    filled_image_bgr = fill_worksheet(str(DEMO_TEX), {}, student_name="Jane Doe")
    filled_image = cv2.cvtColor(filled_image_bgr, cv2.COLOR_BGR2RGB)

    roster = ["Jane Doe", "John Smith", "Alice Johnson", "Bob Lee", "Nancy Drew"]

    assert extract_name(filled_image, boxes["name"], roster) == "Jane Doe"


@pytest.mark.parametrize(
    "response,answer,expected",
    [
        ("123", "123", True),
        (r"\frac{13}{1}", "13", True),
        ("1.234567", "1.23456", True),
        ("123", "128", False),
        (r"\frac{13}{1}", r"\frac{13}{2}", False),
        ("1.234567", "1.28456", False),
    ],
)
def test_is_correct(response, answer, expected):
    assert is_correct(response, answer) == expected


def test_grade_hw_returns_per_question_answer_response_and_correctness(monkeypatch):
    boxes = {"q1": Box(0.1, 0.5, 0.3, 0.05), "q2": Box(0.1, 0.4, 0.3, 0.05)}
    answer_key = {"q1": "12", "q2": "8"}
    responses = {"q1": "12", "q2": "7"}  # q1 right, q2 wrong

    def fake_read_box(image, box):
        qid = next(qid for qid, b in boxes.items() if b is box)
        return responses[qid]

    monkeypatch.setattr("graderbot.read_box", fake_read_box)

    results = grade_hw(answer_key, boxes, np.zeros((10, 10, 3), dtype=np.uint8))

    assert results == {
        "q1": QuestionResult(answer="12", response="12", correct=True),
        "q2": QuestionResult(answer="8", response="7", correct=False),
    }


def test_grade_hw_marks_blank_response_incorrect(monkeypatch):
    boxes = {"q1": Box(0.1, 0.5, 0.3, 0.05)}
    monkeypatch.setattr("graderbot.read_box", lambda image, box: "")

    results = grade_hw({"q1": "12"}, boxes, np.zeros((10, 10, 3), dtype=np.uint8))

    assert results == {"q1": QuestionResult(answer="12", response="", correct=False)}


def test_box_pixel_rect_full_page_box_covers_whole_image():
    assert box_pixel_rect(Box(0.0, 0.0, 1.0, 1.0), 100, 200) == (0, 0, 100, 200)


def test_box_pixel_rect_maps_relative_box_with_y_flipped():
    # Box origin is bottom-left; image origin is top-left, so y is flipped.
    rect = box_pixel_rect(Box(0.25, 0.5, 0.5, 0.25), 100, 200)
    assert rect == (25, 50, 75, 100)


def _make_canonical_worksheet_image(dpi):
    """A white letter page with the four corner aruco markers placed exactly
    where gbworksheet.sty puts them (so their centers are canonical)."""
    width = round(8.5 * dpi)
    height = round(11 * dpi)
    size_px = round(0.75 * dpi)
    inset_px = round(0.15 * dpi)
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    corners = {
        0: (inset_px, inset_px),
        1: (width - inset_px - size_px, inset_px),
        2: (inset_px, height - inset_px - size_px),
        3: (width - inset_px - size_px, height - inset_px - size_px),
    }
    for marker_id, (px, py) in corners.items():
        marker = cv2.aruco.generateImageMarker(_ARUCO_DICTIONARY, marker_id, size_px)
        image[py:py + size_px, px:px + size_px] = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)
    return image


def test_rectify_to_canonical_maps_markers_to_canonical_positions():
    dpi = 100
    canonical = _make_canonical_worksheet_image(dpi)
    height, width = canonical.shape[:2]

    # Simulate a skewed photo via a perspective warp of the canonical page.
    src = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
    inset = 0.03 * min(width, height)
    dst = np.array(
        [
            [inset, 0.5 * inset],
            [width - 1.5 * inset, inset],
            [0.7 * inset, height - inset],
            [width - inset, height - 1.2 * inset],
        ],
        dtype=np.float32,
    )
    skewed = cv2.warpPerspective(
        canonical, cv2.getPerspectiveTransform(src, dst), (width, height), borderValue=(255, 255, 255)
    )

    rectified = rectify_to_canonical(skewed, dpi=dpi)

    centers = _detect_marker_centers(rectified, "rectified")
    expected = _canonical_marker_centers(rectified.shape[1], rectified.shape[0], dpi)
    for marker_id in (0, 1, 2, 3):
        assert centers[marker_id][0] == pytest.approx(expected[marker_id][0], abs=3.0)
        assert centers[marker_id][1] == pytest.approx(expected[marker_id][1], abs=3.0)


def test_align_document_image_corrects_perspective_warp(warped_photo_png):
    aligned_image = align_document_image(str(warped_photo_png), str(DEMO_ANSWER_KEY_PDF))

    reference_image = _render_pdf_page(DEMO_ANSWER_KEY_PDF, dpi=150)

    aligned_centers = _marker_relative_centers(aligned_image)
    reference_centers = _marker_relative_centers(reference_image)

    for marker_id in _MARKER_IDS:
        aligned_x, aligned_y = aligned_centers[marker_id]
        reference_x, reference_y = reference_centers[marker_id]
        assert aligned_x == pytest.approx(reference_x, abs=0.01)
        assert aligned_y == pytest.approx(reference_y, abs=0.01)
