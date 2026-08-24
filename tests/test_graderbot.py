import os
import shutil
import subprocess
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytest

from graderbot.answer_reader import NoOcrAnswerReader
from graderbot.grading import grade_hw, grade_response, is_correct
from graderbot.imaging import (
    _BORDER_MARGIN_PX,
    _crop_box,
    box_pixel_rect,
    crop_box_content_aware,
    ink_fraction,
    is_blank,
    load_pdf_pages_rgb,
    load_scan_pages,
    render_pdf_page_image,
)
from graderbot.models import Box, QuestionResult
from graderbot.ocr import OcrResult, extract_name, read_box
from graderbot.registration import (
    _MARKER_INSET_IN,
    _MARKER_SIZE_IN,
    _canonical_marker_centers,
    _detect_marker_centers,
    align_document_image,
    rectify_to_canonical,
)
from graderbot.worksheet_boxes import (
    extract_answer_boxes,
    extract_answer_boxes_by_page,
)
from graderbot.worksheet_synth import fill_worksheet

DEMO_TEX = Path(__file__).parent.parent / "tex" / "demo.tex"

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
def warped_photo_png(tmp_path_factory, demo_pdf):
    reference_image = _render_pdf_page(demo_pdf, dpi=150)
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


@pytest.mark.slow
def test_extract_answer_boxes_finds_all_ids(boxes):
    assert set(boxes) == {"name", "add001", "sub001", "frac001"}


@pytest.mark.slow
def test_extract_answer_boxes_coordinates_are_relative(boxes):
    for box in boxes.values():
        assert 0 <= box.x_lower_left <= 1
        assert 0 <= box.y_lower_left <= 1
        assert 0 < box.width <= 1
        assert 0 < box.height <= 1


@pytest.mark.slow
def test_extract_answer_boxes_sub001_is_below_add001(boxes):
    assert boxes["sub001"].y_lower_left < boxes["add001"].y_lower_left


@pytest.mark.slow
def test_extract_answer_boxes_by_page_single_page_demo(demo_pdf):
    """demo.tex is one page, so the per-page result is a one-element list whose
    only dict matches the flat `extract_answer_boxes` mapping."""
    pages = extract_answer_boxes_by_page(str(demo_pdf))

    assert len(pages) == 1
    assert set(pages[0]) == {"name", "add001", "sub001", "frac001"}
    assert pages[0] == extract_answer_boxes(str(demo_pdf))


def test_load_pdf_pages_rgb_returns_every_page(tmp_path):
    from PIL import Image

    pages = [
        Image.fromarray(np.full((40, 30, 3), fill, dtype=np.uint8))
        for fill in (50, 150, 250)
    ]
    pdf_path = tmp_path / "multi.pdf"
    pages[0].save(pdf_path, "PDF", save_all=True, append_images=pages[1:])

    loaded = load_pdf_pages_rgb(str(pdf_path))

    assert len(loaded) == 3
    assert all(page.ndim == 3 and page.shape[2] == 3 for page in loaded)


def test_load_pdf_pages_rgb_rasterizes_lazily(tmp_path, monkeypatch):
    """`len()` must be cheap (no rasterizing), and pages should only be
    rasterized as they're actually accessed -- otherwise a large multi-page
    scan holds every page in memory at once before any processing starts,
    which is what OOM'd fly.io on a 15MB roster upload (issue #52)."""
    from PIL import Image

    from graderbot import imaging

    pages = [
        Image.fromarray(np.full((40, 30, 3), fill, dtype=np.uint8))
        for fill in (50, 150, 250)
    ]
    pdf_path = tmp_path / "multi.pdf"
    pages[0].save(pdf_path, "PDF", save_all=True, append_images=pages[1:])

    render_calls = []
    original_render = imaging.render_pdf_page_image

    def _tracking_render(pdf_filename, dpi=imaging._WORKSHEET_RENDER_DPI, page_index=0):
        render_calls.append(page_index)
        return original_render(pdf_filename, dpi, page_index)

    monkeypatch.setattr(imaging, "render_pdf_page_image", _tracking_render)

    loaded = load_pdf_pages_rgb(str(pdf_path))
    assert len(loaded) == 3
    assert render_calls == []  # len() alone must not rasterize anything

    it = iter(loaded)
    next(it)
    assert render_calls == [0]  # only the first page has been rasterized so far


def test_render_pdf_page_image_caps_runaway_page_geometry(tmp_path):
    """Some scanning apps stuff a photo's raw pixel dimensions into the PDF
    page's point-based MediaBox instead of a real physical size -- PIL's
    default PDF save (no `resolution` given) does exactly this, treating each
    pixel as one point. Rendering that at our normal dpi would blow up to a
    multi-hundred-MB raster and OOM'd fly.io on a real scan (issue #52), so
    the output must be capped regardless of what the page claims its size is."""
    from PIL import Image

    from graderbot.imaging import _MAX_RENDER_DIMENSION_PX

    huge_page = Image.fromarray(np.full((3000, 2500, 3), 200, dtype=np.uint8))
    pdf_path = tmp_path / "huge.pdf"
    huge_page.save(pdf_path, "PDF")  # MediaBox ends up 2500pt x 3000pt

    image = render_pdf_page_image(str(pdf_path))

    assert max(image.shape[:2]) <= _MAX_RENDER_DIMENSION_PX


def test_load_scan_pages_reads_pdf_pages_and_single_raster(tmp_path):
    from PIL import Image

    two_pages = [Image.new("RGB", (30, 40), color) for color in ("black", "white")]
    pdf_path = tmp_path / "scans.pdf"
    two_pages[0].save(pdf_path, "PDF", save_all=True, append_images=two_pages[1:])
    assert len(load_scan_pages(str(pdf_path))) == 2

    png_path = tmp_path / "scan.png"
    cv2.imwrite(str(png_path), np.full((40, 30, 3), 200, dtype=np.uint8))
    assert len(load_scan_pages(str(png_path))) == 1


@pytest.mark.slow
def test_read_box_reads_handwritten_answers(boxes):
    if not os.environ.get("MATHPIX_APP_ID") or not os.environ.get("MATHPIX_APP_KEY"):
        pytest.skip("Mathpix credentials are not configured")
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk is not installed")

    # Render the answers at runtime from the current worksheet layout rather
    # than reading a static fixture: `read_box` applies the box coordinates
    # (extracted from a fresh compile) directly to this image with no marker
    # alignment, so a checked-in PDF silently drifts out of sync whenever the
    # layout changes. See the sibling fraction test for the same pattern.
    filled_image_bgr = fill_worksheet(str(DEMO_TEX), {"add001": "12", "sub001": "11"})[0]
    filled_image = cv2.cvtColor(filled_image_bgr, cv2.COLOR_BGR2RGB)

    assert read_box(filled_image, boxes["add001"]).text == "12"
    assert read_box(filled_image, boxes["sub001"]).text == "11"


@pytest.mark.slow
def test_read_box_reads_a_handwritten_fraction(boxes):
    if not os.environ.get("MATHPIX_APP_ID") or not os.environ.get("MATHPIX_APP_KEY"):
        pytest.skip("Mathpix credentials are not configured")
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk is not installed")

    filled_image_bgr = fill_worksheet(str(DEMO_TEX), {"frac001": r"\frac{3}{4}"})[0]
    filled_image = cv2.cvtColor(filled_image_bgr, cv2.COLOR_BGR2RGB)

    assert read_box(filled_image, boxes["frac001"]).text == r"\frac{3}{4}"


@pytest.mark.slow
def test_extract_name_matches_closest_roster_name(boxes):
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract is not installed")
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk is not installed")

    filled_image_bgr = fill_worksheet(str(DEMO_TEX), {}, student_name="Jane Doe")[0]
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
        # Unsimplified fractions with the right value are wrong (issue #39).
        (r"\frac{10}{21}", r"\frac{10}{21}", True),
        (r"\frac{20}{42}", r"\frac{10}{21}", False),
        (r"\frac{5}{10}", r"\frac{1}{2}", False),
        (r"\frac{26}{2}", "13", False),
    ],
)
def test_is_correct(response, answer, expected):
    assert is_correct(response, answer) == expected


@pytest.mark.parametrize(
    "response,answer,expected",
    [
        # Issue #39: mathpix reads the fraction bar as a '1'.
        ("10 1 21", r"\frac{10}{21}", True),
        # Literal forward slash, with and without spaces.
        ("10 / 21", r"\frac{10}{21}", True),
        ("10/21", r"\frac{10}{21}", True),
        # Fully whitespace-collapsed, bar reinterpreted from the middle '1'.
        ("5112", r"\frac{5}{12}", True),
        ("5 1 12", r"\frac{5}{12}", True),
        # Mixed number: 1 3/4 == 7/4, and its bar-as-1 garble "1 3 1 4".
        ("1 3/4", r"\frac{7}{4}", True),
        ("1 3 1 4", r"\frac{7}{4}", True),
        # Unsimplified fractions of the right value must still be wrong.
        ("20 1 42", r"\frac{10}{21}", False),
        ("20/42", r"\frac{10}{21}", False),
        # A genuinely wrong response stays wrong.
        ("10 1 22", r"\frac{10}{21}", False),
        # Leniency must not fire for integer answers.
        ("131", "13", False),
    ],
)
def test_is_correct_tolerates_mathpix_fraction_garble(response, answer, expected):
    assert is_correct(response, answer) == expected


@pytest.mark.parametrize(
    "response,answer,expected",
    [
        # Right value but not in lowest terms: still wrong (issue #38); issue
        # #71 dropped the "simplify" feedback note, so it's just wrong now,
        # like any other wrong answer.
        (r"\frac{45}{120}", r"\frac{3}{8}", False),
        (r"\frac{20}{42}", r"\frac{10}{21}", False),
        (r"\frac{5}{10}", r"\frac{1}{2}", False),
        # Improper fraction equal to a whole number is also "not simplified".
        (r"\frac{26}{2}", "13", False),
        # Already reduced and correct.
        (r"\frac{3}{8}", r"\frac{3}{8}", True),
        (r"\frac{13}{1}", "13", True),
        # Wrong value.
        (r"\frac{2}{6}", r"\frac{3}{8}", False),
        (r"\frac{1}{3}", r"\frac{3}{8}", False),
        # Garbled-OCR fraction that is genuinely right stays correct.
        ("10 1 21", r"\frac{10}{21}", True),
        # Garbled OCR that is right in value but unreduced is still wrong.
        ("20 1 42", r"\frac{10}{21}", False),
        ("20/42", r"\frac{10}{21}", False),
        # Genuinely wrong garble stays wrong.
        ("10 1 22", r"\frac{10}{21}", False),
    ],
)
def test_grade_response_flags_unsimplified_fractions(response, answer, expected):
    assert grade_response(response, answer) == expected


def test_grade_hw_returns_per_question_answer_response_and_correctness(monkeypatch):
    boxes = {"q1": Box(0.1, 0.5, 0.3, 0.05), "q2": Box(0.1, 0.4, 0.3, 0.05)}
    answer_key = {"q1": "12", "q2": "8"}
    responses = {"q1": "12", "q2": "7"}  # q1 right, q2 wrong

    def fake_read_box(image, box):
        qid = next(qid for qid, b in boxes.items() if b is box)
        return OcrResult(text=responses[qid], raw_text=responses[qid], confidence=0.9)

    monkeypatch.setattr("graderbot.answer_reader.read_box", fake_read_box)

    results = grade_hw(answer_key, boxes, np.zeros((10, 10, 3), dtype=np.uint8))

    assert results == {
        "q1": QuestionResult(answer="12", response="12", correct=True, ocr_confidence=0.9, ocr_raw="12"),
        "q2": QuestionResult(answer="8", response="7", correct=False, ocr_confidence=0.9, ocr_raw="7"),
    }


def test_grade_hw_marks_blank_response_incorrect(monkeypatch):
    boxes = {"q1": Box(0.1, 0.5, 0.3, 0.05)}
    monkeypatch.setattr(
        "graderbot.answer_reader.read_box",
        lambda image, box: OcrResult(text="", raw_text="", confidence=0.1),
    )

    results = grade_hw({"q1": "12"}, boxes, np.zeros((10, 10, 3), dtype=np.uint8))

    assert results == {
        "q1": QuestionResult(answer="12", response="", correct=False, ocr_confidence=0.1, ocr_raw="")
    }


def test_grade_hw_skips_mathpix_for_a_blank_box(monkeypatch):
    # A realistically-sized white canvas so the inset crop is non-degenerate
    # (unlike the 10x10 canvases above, whose crops round to zero pixels).
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    boxes = {"q1": Box(0.1, 0.5, 0.3, 0.1)}

    def fail_if_called(image, box):
        pytest.fail("read_box (Mathpix) called for a blank box")

    monkeypatch.setattr("graderbot.answer_reader.read_box", fail_if_called)

    results = grade_hw({"q1": "12"}, boxes, image)

    assert results == {"q1": QuestionResult(answer="12", response="", correct=False, blank=True)}


def test_grade_hw_never_grades_an_open_ended_question(monkeypatch):
    # issue #65: an open-ended question's response is captured but never
    # compared against the stored answer, even when it plainly disagrees.
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    boxes = {"q1": Box(0.1, 0.5, 0.3, 0.1)}
    x0, y0, x1, y1 = box_pixel_rect(boxes["q1"], 200, 200)
    cv2.rectangle(image, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (0, 0, 0), -1)
    monkeypatch.setattr(
        "graderbot.answer_reader.read_box",
        lambda image, box: OcrResult(text="I like fractions", raw_text="I like fractions", confidence=0.8),
    )

    results = grade_hw({"q1": ""}, boxes, image, open_ended={"q1": True})

    assert results == {
        "q1": QuestionResult(
            answer="",
            response="I like fractions",
            correct=False,
            open_ended=True,
            ocr_confidence=0.8,
            ocr_raw="I like fractions",
        )
    }


def test_grade_hw_skips_mathpix_for_a_blank_open_ended_box(monkeypatch):
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    boxes = {"q1": Box(0.1, 0.5, 0.3, 0.1)}

    def fail_if_called(image, box):
        pytest.fail("read_box (Mathpix) called for a blank box")

    monkeypatch.setattr("graderbot.answer_reader.read_box", fail_if_called)

    results = grade_hw({"q1": ""}, boxes, image, open_ended={"q1": True})

    assert results == {
        "q1": QuestionResult(answer="", response="", correct=False, blank=True, open_ended=True)
    }


def test_grade_hw_reads_a_box_with_ink_normally(monkeypatch):
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    boxes = {"q1": Box(0.1, 0.5, 0.3, 0.1)}
    # Draw ink inside the box's pixel rect so it clears the blank threshold.
    x0, y0, x1, y1 = box_pixel_rect(boxes["q1"], 200, 200)
    cv2.rectangle(image, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (0, 0, 0), -1)
    monkeypatch.setattr(
        "graderbot.answer_reader.read_box",
        lambda image, box: OcrResult(text="12", raw_text="12", confidence=0.95),
    )

    results = grade_hw({"q1": "12"}, boxes, image)

    assert results == {
        "q1": QuestionResult(
            answer="12", response="12", correct=True, blank=False, ocr_confidence=0.95, ocr_raw="12"
        )
    }


def test_grade_hw_no_ocr_reader_marks_a_filled_box_wrong_without_reading_it(monkeypatch):
    # issue #83: a filled-in box is marked wrong -- not blank -- so markup
    # still shows the student the correct answer, but no OCR backend is ever
    # asked to transcribe the handwriting.
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    boxes = {"q1": Box(0.1, 0.5, 0.3, 0.1)}
    x0, y0, x1, y1 = box_pixel_rect(boxes["q1"], 200, 200)
    cv2.rectangle(image, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (0, 0, 0), -1)
    monkeypatch.setattr(
        "graderbot.answer_reader.read_box",
        lambda image, box: pytest.fail("Mathpix must not be called under NoOcrAnswerReader"),
    )

    results = grade_hw({"q1": "12"}, boxes, image, answer_reader=NoOcrAnswerReader())

    assert results == {
        "q1": QuestionResult(
            answer="12", response="", correct=False, blank=False, ocr_raw="", ocr_source="no_ocr"
        )
    }


def test_grade_hw_no_ocr_reader_still_detects_a_blank_box(monkeypatch):
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    boxes = {"q1": Box(0.1, 0.5, 0.3, 0.1)}

    results = grade_hw({"q1": "12"}, boxes, image, answer_reader=NoOcrAnswerReader())

    assert results == {"q1": QuestionResult(answer="12", response="", correct=False, blank=True)}


class _FakeResponseScorer:
    """Stands in for `response_scorer.CnnResponseScorer` (issue #81):
    ignores the crop/candidates and always reports `response`, so these
    tests exercise `grade_hw`'s wiring without a trained model."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def score(self, image, box, candidates):
        self.calls.append(candidates)
        from graderbot.response_scorer import ScoredResponse

        return ScoredResponse(best=self.response, scores={self.response: 0.7}, log_probs={self.response: -1.0})


def test_grade_hw_uses_response_scorer_for_a_plain_numeric_answer(monkeypatch):
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    boxes = {"q1": Box(0.1, 0.5, 0.3, 0.1)}
    x0, y0, x1, y1 = box_pixel_rect(boxes["q1"], 200, 200)
    cv2.rectangle(image, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (0, 0, 0), -1)

    def fail_if_called(image, box):
        pytest.fail("answer_reader called even though response_scorer handles plain-numeric answers")

    monkeypatch.setattr("graderbot.answer_reader.read_box", fail_if_called)
    scorer = _FakeResponseScorer("12")

    results = grade_hw({"q1": "12"}, boxes, image, response_scorer=scorer)

    assert results == {
        "q1": QuestionResult(
            answer="12", response="12", correct=True, ocr_confidence=0.7, ocr_raw="12", ocr_source="response_scorer"
        )
    }
    assert scorer.calls[0][0] == "12"  # candidates[0] is always the stored answer


def test_grade_hw_falls_back_to_answer_reader_for_a_fraction_answer(monkeypatch):
    # response_scorer's v1 scope is plain numeric only (response_candidates
    # module docstring) -- a fraction answer must still go to answer_reader
    # even when a response_scorer is supplied.
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    boxes = {"q1": Box(0.1, 0.5, 0.3, 0.1)}
    x0, y0, x1, y1 = box_pixel_rect(boxes["q1"], 200, 200)
    cv2.rectangle(image, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (0, 0, 0), -1)
    monkeypatch.setattr(
        "graderbot.answer_reader.read_box",
        lambda image, box: OcrResult(text="\\frac{3}{4}", raw_text="\\frac{3}{4}", confidence=0.9),
    )
    scorer = _FakeResponseScorer("should not be used")

    results = grade_hw({"q1": r"\frac{3}{4}"}, boxes, image, response_scorer=scorer)

    assert scorer.calls == []
    assert results["q1"].response == "\\frac{3}{4}"


def test_ink_fraction_is_zero_for_a_blank_image():
    blank = np.full((20, 20, 3), 255, dtype=np.uint8)
    assert ink_fraction(blank) == 0.0


def test_ink_fraction_counts_dark_pixels():
    image = np.full((20, 20, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (19, 19), (0, 0, 0), -1)  # fully inked
    assert ink_fraction(image) == pytest.approx(1.0)


def test_is_blank_true_for_a_mostly_white_image():
    blank = np.full((20, 20, 3), 255, dtype=np.uint8)
    assert is_blank(blank)


def test_is_blank_false_once_ink_exceeds_the_threshold():
    image = np.full((20, 20, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (19, 19), (0, 0, 0), -1)
    assert not is_blank(image)


def test_is_blank_false_for_faint_pencil_ink():
    # issue #74: a real scanned answer written in faint pencil never actually
    # goes near-black -- graphite scans as mid-gray, not ink-black. A crop from
    # the reported scan had answer pixels sitting mostly in the 128-220 gray
    # range and covering ~3% of the box, yet the old darkness cutoff (128)
    # counted almost none of them as "ink", so a genuinely answered box read as
    # blank and skipped Mathpix entirely. Simulate that here with a small gray
    # (170) rectangle rather than pure black.
    image = np.full((50, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (30, 40), (170, 170, 170), -1)  # ~3% of the box
    assert not is_blank(image)


def _draw_box_border(image, box, image_width, image_height, thickness=1):
    """Draws just the printed rectangle outline for `box` -- the border
    `crop_box_content_aware` must exclude, distinct from the ink inside it."""
    x0, y0, x1, y1 = box_pixel_rect(box, image_width, image_height)
    cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 0), thickness)


def test_crop_box_content_aware_falls_back_to_fixed_inset_when_blank():
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    box = Box(0.1, 0.5, 0.3, 0.2)
    _draw_box_border(image, box, 200, 200)

    result = crop_box_content_aware(image, box, fallback_inset=0.08)

    assert np.array_equal(result, _crop_box(image, box, 0.08))


def test_crop_box_content_aware_excludes_the_border():
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    box = Box(0.1, 0.5, 0.3, 0.2)
    _draw_box_border(image, box, 200, 200)
    x0, y0, x1, y1 = box_pixel_rect(box, 200, 200)
    cv2.rectangle(image, (x0 + 15, y0 + 10), (x0 + 25, y0 + 20), (0, 0, 0), -1)  # ink well inside

    result = crop_box_content_aware(image, box, fallback_inset=0.08)

    # None of the four 1px-wide border edges made it into the crop.
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    assert (gray[0, :] > 200).all()
    assert (gray[-1, :] > 200).all()
    assert (gray[:, 0] > 200).all()
    assert (gray[:, -1] > 200).all()


def test_crop_box_content_aware_keeps_ink_written_flush_against_an_edge():
    # Reproduces issue #70's follow-up: a multi-digit answer written flush
    # against the box's left edge (as opposed to centered, with margin to
    # spare) had its leading digit sliced off entirely by the old
    # fixed-fraction inset, because that fraction is computed from the box's
    # own (wide) width rather than the border's actual thickness.
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    box = Box(0.1, 0.5, 0.6, 0.15)  # a wide box, like a real answer box
    _draw_box_border(image, box, 200, 200)
    x0, y0, x1, y1 = box_pixel_rect(box, 200, 200)
    # A vertical stroke just past the border -- inside the fixed-fraction
    # inset's dead zone, but real content, not border.
    stroke_x = x0 + _BORDER_MARGIN_PX + 2
    cv2.line(image, (stroke_x, y0 + 5), (stroke_x, y1 - 5), (0, 0, 0), 2)

    fixed_inset_crop = _crop_box(image, box, 0.08)
    fixed_inset_gray = cv2.cvtColor(fixed_inset_crop, cv2.COLOR_RGB2GRAY)
    assert (fixed_inset_gray < 200).sum() == 0  # confirms the old behavior clips it

    result = crop_box_content_aware(image, box, fallback_inset=0.08)
    result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    assert (result_gray < 200).any()  # the new crop keeps it


def test_crop_box_content_aware_pads_around_a_tight_answer():
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    box = Box(0.1, 0.5, 0.4, 0.2)
    _draw_box_border(image, box, 200, 200)
    x0, y0, x1, y1 = box_pixel_rect(box, 200, 200)
    ink_w, ink_h = 10, 8
    cv2.rectangle(image, (x0 + 20, y0 + 20), (x0 + 20 + ink_w, y0 + 20 + ink_h), (0, 0, 0), -1)

    result = crop_box_content_aware(image, box, fallback_inset=0.08)

    # Tighter than the whole box, but bigger than the bare ink -- padding was added.
    assert ink_h < result.shape[0] < (y1 - y0)
    assert ink_w < result.shape[1] < (x1 - x0)


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
    size_px = round(_MARKER_SIZE_IN * dpi)
    inset_px = round(_MARKER_INSET_IN * dpi)
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


def test_markers_survive_printer_margin_clip():
    """The marker inset must keep the fiducials clear of a printer's
    non-printable margin: even after the outer 0.25in of the page is clipped
    (whitened), all four markers must still decode. Guards against regressing
    the inset back toward the paper edge, which clipped marker borders and
    defeated detection on real printouts (issue #35)."""
    dpi = 150
    clip_in = 0.25
    canonical = _make_canonical_worksheet_image(dpi)

    clipped = canonical.copy()
    m = round(clip_in * dpi)
    clipped[:m, :] = 255
    clipped[-m:, :] = 255
    clipped[:, :m] = 255
    clipped[:, -m:] = 255

    centers = _detect_marker_centers(clipped, "clipped")
    assert set(centers) == {0, 1, 2, 3}


@pytest.mark.slow
def test_align_document_image_corrects_perspective_warp(warped_photo_png, demo_pdf):
    aligned_image = align_document_image(str(warped_photo_png), str(demo_pdf))

    reference_image = _render_pdf_page(demo_pdf, dpi=150)

    aligned_centers = _marker_relative_centers(aligned_image)
    reference_centers = _marker_relative_centers(reference_image)

    for marker_id in _MARKER_IDS:
        aligned_x, aligned_y = aligned_centers[marker_id]
        reference_x, reference_y = reference_centers[marker_id]
        assert aligned_x == pytest.approx(reference_x, abs=0.01)
        assert aligned_y == pytest.approx(reference_y, abs=0.01)
