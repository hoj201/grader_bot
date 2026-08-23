"""Tests for the name-box OCR helpers (issue #58) and the Mathpix answer-box
OCR request/cleanup (alphabets_allowed + Greek-letter misread fix) and result
(OcrResult text/raw_text/confidence, issue #70).

These monkeypatch the Tesseract/Mathpix calls so they run without the binary
or a live API key; the end-to-end read of a real rendered worksheet lives in
test_graderbot.py.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from graderbot import ocr
from graderbot.models import Box
from graderbot.ocr import _fix_greek_misreads, extract_name, extract_name_scored

FULL_BOX = Box(x_lower_left=0.0, y_lower_left=0.0, width=1.0, height=1.0)
ROSTER = ["Jane Doe", "John Smith", "Alice Johnson"]


@pytest.fixture
def blank_page():
    return np.full((120, 400, 3), 255, np.uint8)


def test_extract_name_scored_returns_match_and_similarity(monkeypatch, blank_page):
    # A typical Tesseract misread of handwriting: a couple of letters wrong.
    monkeypatch.setattr(ocr, "_tesseract_ocr_name", lambda image: "Jane Doo")

    name, score = extract_name_scored(blank_page, FULL_BOX, ROSTER)

    assert name == "Jane Doe"
    assert 0.0 < score < 1.0


def test_extract_name_scored_gives_a_clean_read_full_confidence(monkeypatch, blank_page):
    monkeypatch.setattr(ocr, "_tesseract_ocr_name", lambda image: "Jane Doe")

    assert extract_name_scored(blank_page, FULL_BOX, ROSTER) == ("Jane Doe", 1.0)


def test_extract_name_scored_returns_empty_when_nothing_is_close(monkeypatch, blank_page):
    monkeypatch.setattr(ocr, "_tesseract_ocr_name", lambda image: "zzzzzzzzzz")

    assert extract_name_scored(blank_page, FULL_BOX, ROSTER) == ("", 0.0)


def test_extract_name_scored_ranks_a_cleaner_read_higher(monkeypatch, blank_page):
    monkeypatch.setattr(ocr, "_tesseract_ocr_name", lambda image: "Jane Doo")
    _, close = extract_name_scored(blank_page, FULL_BOX, ROSTER)
    monkeypatch.setattr(ocr, "_tesseract_ocr_name", lambda image: "Jarie Dae")
    _, far = extract_name_scored(blank_page, FULL_BOX, ROSTER)

    assert close > far


def test_extract_name_returns_just_the_name(monkeypatch, blank_page):
    """The scoreless wrapper still behaves exactly as before."""
    monkeypatch.setattr(ocr, "_tesseract_ocr_name", lambda image: "Jhn Smith")

    assert extract_name(blank_page, FULL_BOX, ROSTER) == "John Smith"


def test_extract_name_returns_empty_string_on_no_match(monkeypatch, blank_page):
    monkeypatch.setattr(ocr, "_tesseract_ocr_name", lambda image: "zzzzzzzzzz")

    assert extract_name(blank_page, FULL_BOX, ROSTER) == ""


def test_fix_greek_misreads_replaces_alpha_with_two():
    assert _fix_greek_misreads(r"\alpha") == "2"


def test_fix_greek_misreads_replaces_alpha_inside_a_fraction():
    assert _fix_greek_misreads(r"\frac{\alpha}{3}") == r"\frac{2}{3}"


def test_fix_greek_misreads_leaves_plain_digits_untouched():
    assert _fix_greek_misreads(r"\frac{3}{4}") == r"\frac{3}{4}"


def _mock_mathpix_response(text: str, confidence=0.95) -> MagicMock:
    response = MagicMock()
    payload = {"text": text}
    if confidence is not None:
        payload["confidence"] = confidence
    response.json.return_value = payload
    return response


def test_mathpix_ocr_does_not_request_alphabets_allowed(monkeypatch):
    """Regression guard for the alphabets_allowed experiment: real testing
    showed it makes Mathpix refuse to guess on legible handwritten math
    (see the comment above _GREEK_MISREAD_MAP in ocr.py), so it must not be
    sent."""
    monkeypatch.setenv("MATHPIX_APP_ID", "test-id")
    monkeypatch.setenv("MATHPIX_APP_KEY", "test-key")
    monkeypatch.delenv("MATHPIX_LOG_BUCKET", raising=False)
    monkeypatch.delenv("S3_BUCKET", raising=False)
    image = np.full((40, 40, 3), 255, np.uint8)

    with patch(
        "graderbot.ocr.requests.post", return_value=_mock_mathpix_response("12")
    ) as mock_post:
        ocr._mathpix_ocr(image)

    _, kwargs = mock_post.call_args
    assert "alphabets_allowed" not in kwargs["json"]


def test_mathpix_ocr_fixes_greek_misreads_in_the_response(monkeypatch):
    monkeypatch.setenv("MATHPIX_APP_ID", "test-id")
    monkeypatch.setenv("MATHPIX_APP_KEY", "test-key")
    monkeypatch.delenv("MATHPIX_LOG_BUCKET", raising=False)
    monkeypatch.delenv("S3_BUCKET", raising=False)
    image = np.full((40, 40, 3), 255, np.uint8)

    with patch(
        "graderbot.ocr.requests.post", return_value=_mock_mathpix_response(r"\alpha")
    ):
        result = ocr._mathpix_ocr(image)

    # `text` is repaired for grading; `raw_text` keeps what Mathpix actually
    # said, so a misread can still be diagnosed (issue #70).
    assert result.text == "2"
    assert result.raw_text == r"\alpha"


def test_mathpix_ocr_returns_mathpix_confidence(monkeypatch):
    monkeypatch.setenv("MATHPIX_APP_ID", "test-id")
    monkeypatch.setenv("MATHPIX_APP_KEY", "test-key")
    monkeypatch.delenv("MATHPIX_LOG_BUCKET", raising=False)
    monkeypatch.delenv("S3_BUCKET", raising=False)
    image = np.full((40, 40, 3), 255, np.uint8)

    with patch(
        "graderbot.ocr.requests.post",
        return_value=_mock_mathpix_response("12", confidence=0.42),
    ):
        result = ocr._mathpix_ocr(image)

    assert result.confidence == 0.42


def test_mathpix_ocr_tags_the_result_with_its_source(monkeypatch):
    """So a graded result can be traced back to which OCR backend produced
    it once more than one is available (issue #70)."""
    monkeypatch.setenv("MATHPIX_APP_ID", "test-id")
    monkeypatch.setenv("MATHPIX_APP_KEY", "test-key")
    monkeypatch.delenv("MATHPIX_LOG_BUCKET", raising=False)
    monkeypatch.delenv("S3_BUCKET", raising=False)
    image = np.full((40, 40, 3), 255, np.uint8)

    with patch(
        "graderbot.ocr.requests.post", return_value=_mock_mathpix_response("12")
    ):
        result = ocr._mathpix_ocr(image)

    assert result.source == ocr.MATHPIX_SOURCE == "mathpix"


def test_mathpix_ocr_confidence_is_none_when_mathpix_omits_it(monkeypatch):
    monkeypatch.setenv("MATHPIX_APP_ID", "test-id")
    monkeypatch.setenv("MATHPIX_APP_KEY", "test-key")
    monkeypatch.delenv("MATHPIX_LOG_BUCKET", raising=False)
    monkeypatch.delenv("S3_BUCKET", raising=False)
    image = np.full((40, 40, 3), 255, np.uint8)

    with patch(
        "graderbot.ocr.requests.post",
        return_value=_mock_mathpix_response("12", confidence=None),
    ):
        result = ocr._mathpix_ocr(image)

    assert result.confidence is None
