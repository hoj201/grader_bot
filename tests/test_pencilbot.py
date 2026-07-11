import os
import shutil
import subprocess
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytest

from pencilbot import align_document_image, extract_answer_boxes, extract_name, read_box
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


def test_extract_answer_boxes_finds_all_ids(demo_pdf):
    boxes = extract_answer_boxes(str(demo_pdf))
    assert set(boxes) == {"name", "add001", "sub001", "frac001"}


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


def test_read_box_reads_handwritten_answers(demo_pdf):
    if not os.environ.get("MATHPIX_APP_ID") or not os.environ.get("MATHPIX_APP_KEY"):
        pytest.skip("Mathpix credentials are not configured")

    boxes = extract_answer_boxes(str(demo_pdf))

    assert read_box(str(DEMO_ANSWER_KEY_PDF), boxes["add001"]) == "12"
    assert read_box(str(DEMO_ANSWER_KEY_PDF), boxes["sub001"]) == "11"


def test_read_box_reads_a_handwritten_fraction(demo_pdf, tmp_path):
    if not os.environ.get("MATHPIX_APP_ID") or not os.environ.get("MATHPIX_APP_KEY"):
        pytest.skip("Mathpix credentials are not configured")
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk is not installed")

    boxes = extract_answer_boxes(str(demo_pdf))

    filled_image = fill_worksheet(str(DEMO_TEX), {"frac001": r"\frac{3}{4}"})
    filled_fn = tmp_path / "filled_frac.png"
    cv2.imwrite(str(filled_fn), filled_image)

    assert read_box(str(filled_fn), boxes["frac001"]) == r"\frac{3}{4}"


def test_extract_name_matches_closest_roster_name(demo_pdf, tmp_path):
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract is not installed")
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk is not installed")

    boxes = extract_answer_boxes(str(demo_pdf))

    filled_image = fill_worksheet(str(DEMO_TEX), {}, student_name="Jane Doe")
    filled_fn = tmp_path / "filled_name.png"
    cv2.imwrite(str(filled_fn), filled_image)

    roster = ["Jane Doe", "John Smith", "Alice Johnson", "Bob Lee", "Nancy Drew"]

    assert extract_name(str(filled_fn), boxes["name"], roster) == "Jane Doe"


def test_align_document_image_corrects_perspective_warp(warped_photo_png):
    aligned_filename = align_document_image(
        str(warped_photo_png), str(DEMO_ANSWER_KEY_PDF)
    )

    try:
        aligned_image = cv2.imread(aligned_filename)
        assert aligned_image is not None

        reference_image = _render_pdf_page(DEMO_ANSWER_KEY_PDF, dpi=150)

        aligned_centers = _marker_relative_centers(aligned_image)
        reference_centers = _marker_relative_centers(reference_image)

        for marker_id in _MARKER_IDS:
            aligned_x, aligned_y = aligned_centers[marker_id]
            reference_x, reference_y = reference_centers[marker_id]
            assert aligned_x == pytest.approx(reference_x, abs=0.01)
            assert aligned_y == pytest.approx(reference_y, abs=0.01)
    finally:
        Path(aligned_filename).unlink(missing_ok=True)
