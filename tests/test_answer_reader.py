"""Tests for the three AnswerReader strategies (issue #70): MathpixAnswerReader
(a thin wrapper over ocr.read_box), EasyOcrAnswerReader (an HTTP client for the
easyocr_service sidecar), and GoogleVisionAnswerReader (a direct REST call to
the Vision API). All three are mocked at the request boundary -- no real
Mathpix/Vision API or easyocr_service container is needed.
"""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import requests

from graderbot import answer_reader
from graderbot.answer_reader import (
    EASYOCR_DEFAULT_ALLOWLIST,
    EASYOCR_SOURCE,
    GOOGLE_VISION_SOURCE,
    EasyOcrAnswerReader,
    GoogleVisionAnswerReader,
    MathpixAnswerReader,
    _detect_fraction_bar,
)
from graderbot.models import Box
from graderbot.ocr import OcrResult

BOX = Box(x_lower_left=0.1, y_lower_left=0.1, width=0.3, height=0.1)


def test_mathpix_answer_reader_delegates_to_read_box(monkeypatch):
    sentinel = OcrResult(text="12", raw_text="12", confidence=0.9)
    monkeypatch.setattr(answer_reader, "read_box", lambda image, box: sentinel)

    image = np.full((200, 200, 3), 255, np.uint8)
    assert MathpixAnswerReader().read(image, BOX) is sentinel


def test_easyocr_answer_reader_requires_a_service_url(monkeypatch):
    monkeypatch.delenv("EASYOCR_SERVICE_URL", raising=False)

    with pytest.raises(EnvironmentError):
        EasyOcrAnswerReader()


def test_easyocr_answer_reader_uses_service_url_env_var(monkeypatch):
    monkeypatch.setenv("EASYOCR_SERVICE_URL", "http://localhost:8080")

    assert EasyOcrAnswerReader().service_url == "http://localhost:8080"


def test_easyocr_answer_reader_defaults_to_digits_and_dot(monkeypatch):
    monkeypatch.setenv("EASYOCR_SERVICE_URL", "http://localhost:8080")

    assert EasyOcrAnswerReader().allowlist == EASYOCR_DEFAULT_ALLOWLIST == "0123456789."


def _mock_ocr_response(text: str, confidence) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"text": text, "confidence": confidence}
    return response


def test_easyocr_answer_reader_posts_image_and_allowlist(monkeypatch):
    reader = EasyOcrAnswerReader(allowlist="0123456789.xy", service_url="http://localhost:8080")
    image = np.full((200, 200, 3), 255, np.uint8)

    with patch(
        "graderbot.answer_reader.requests.post",
        return_value=_mock_ocr_response("14", 0.87),
    ) as mock_post:
        result = reader.read(image, BOX)

    url, kwargs = mock_post.call_args
    assert url[0] == "http://localhost:8080/ocr"
    assert kwargs["json"]["allowlist"] == "0123456789.xy"
    assert kwargs["json"]["image"].startswith("data:image/png;base64,")
    assert result == OcrResult(text="14", raw_text="14", confidence=0.87, source=EASYOCR_SOURCE)


def test_easyocr_answer_reader_strips_trailing_slash_from_service_url(monkeypatch):
    reader = EasyOcrAnswerReader(service_url="http://localhost:8080/")
    image = np.full((200, 200, 3), 255, np.uint8)

    with patch(
        "graderbot.answer_reader.requests.post",
        return_value=_mock_ocr_response("7", 0.5),
    ) as mock_post:
        reader.read(image, BOX)

    url, _ = mock_post.call_args
    assert url[0] == "http://localhost:8080/ocr"


def test_easyocr_answer_reader_sends_no_api_key_header_by_default(monkeypatch):
    monkeypatch.delenv("EASYOCR_API_KEY", raising=False)
    reader = EasyOcrAnswerReader(service_url="http://localhost:8080")
    image = np.full((200, 200, 3), 255, np.uint8)

    with patch(
        "graderbot.answer_reader.requests.post",
        return_value=_mock_ocr_response("7", 0.5),
    ) as mock_post:
        reader.read(image, BOX)

    assert mock_post.call_args.kwargs["headers"] == {}


def test_easyocr_answer_reader_sends_api_key_header_when_configured(monkeypatch):
    reader = EasyOcrAnswerReader(service_url="http://localhost:8080", api_key="secret-value")
    image = np.full((200, 200, 3), 255, np.uint8)

    with patch(
        "graderbot.answer_reader.requests.post",
        return_value=_mock_ocr_response("7", 0.5),
    ) as mock_post:
        reader.read(image, BOX)

    assert mock_post.call_args.kwargs["headers"] == {"X-Api-Key": "secret-value"}


def test_easyocr_answer_reader_uses_api_key_env_var(monkeypatch):
    monkeypatch.setenv("EASYOCR_SERVICE_URL", "http://localhost:8080")
    monkeypatch.setenv("EASYOCR_API_KEY", "from-env")

    assert EasyOcrAnswerReader().api_key == "from-env"


def test_google_vision_answer_reader_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_VISION_API_KEY", raising=False)

    with pytest.raises(EnvironmentError):
        GoogleVisionAnswerReader()


def test_google_vision_answer_reader_uses_api_key_env_var(monkeypatch):
    monkeypatch.setenv("GOOGLE_VISION_API_KEY", "test-key")

    assert GoogleVisionAnswerReader().api_key == "test-key"


def _mock_vision_response(text: str, word_confidences) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {
        "responses": [
            {
                "fullTextAnnotation": {
                    "text": text,
                    "pages": [
                        {
                            "blocks": [
                                {
                                    "paragraphs": [
                                        {
                                            "words": [
                                                {"confidence": c} for c in word_confidences
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    ],
                }
            }
        ]
    }
    return response


def test_google_vision_answer_reader_posts_image_and_parses_text_and_confidence(monkeypatch):
    reader = GoogleVisionAnswerReader(api_key="test-key")
    image = np.full((200, 200, 3), 255, np.uint8)

    with patch(
        "graderbot.answer_reader.requests.post",
        return_value=_mock_vision_response("14\n", [0.95]),
    ) as mock_post:
        result = reader.read(image, BOX)

    args, kwargs = mock_post.call_args
    assert args[0] == "https://vision.googleapis.com/v1/images:annotate"
    # The key travels in a header, not a URL query param, so it never lands
    # in requests' own HTTPError message (or anything else logging the URL).
    assert kwargs["headers"] == {"X-goog-api-key": "test-key"}
    assert "params" not in kwargs
    feature = kwargs["json"]["requests"][0]["features"][0]
    assert feature["type"] == "DOCUMENT_TEXT_DETECTION"
    assert "content" in kwargs["json"]["requests"][0]["image"]
    # Trailing newline Vision includes in fullTextAnnotation.text is stripped.
    assert result == OcrResult(text="14", raw_text="14", confidence=0.95, source=GOOGLE_VISION_SOURCE)


def test_google_vision_answer_reader_confidence_is_the_weakest_word(monkeypatch):
    reader = GoogleVisionAnswerReader(api_key="test-key")
    image = np.full((200, 200, 3), 255, np.uint8)

    with patch(
        "graderbot.answer_reader.requests.post",
        return_value=_mock_vision_response("14", [0.95, 0.40]),
    ):
        result = reader.read(image, BOX)

    assert result.confidence == 0.40


def test_google_vision_answer_reader_confidence_is_none_without_words(monkeypatch):
    reader = GoogleVisionAnswerReader(api_key="test-key")
    image = np.full((200, 200, 3), 255, np.uint8)

    with patch(
        "graderbot.answer_reader.requests.post",
        return_value=_mock_vision_response("", []),
    ):
        result = reader.read(image, BOX)

    assert result.confidence is None


def test_google_vision_answer_reader_raises_on_a_vision_api_error(monkeypatch):
    reader = GoogleVisionAnswerReader(api_key="test-key")
    image = np.full((200, 200, 3), 255, np.uint8)
    response = MagicMock()
    response.json.return_value = {"responses": [{"error": {"message": "Bad image data."}}]}

    with patch("graderbot.answer_reader.requests.post", return_value=response):
        with pytest.raises(RuntimeError, match="Bad image data"):
            reader.read(image, BOX)


def test_google_vision_answer_reader_never_puts_the_key_in_the_url(monkeypatch):
    """Regression guard: an earlier version passed the key as a `?key=...`
    query param, which leaked it into requests' own HTTPError message (and
    anywhere else that logs response.url) on any failed call."""
    reader = GoogleVisionAnswerReader(api_key="super-secret-key")
    image = np.full((200, 200, 3), 255, np.uint8)
    response = MagicMock()
    response.url = "https://vision.googleapis.com/v1/images:annotate"
    response.text = "Forbidden"
    response.json.return_value = {"error": {"message": "Forbidden"}}
    response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "403 Client Error: Forbidden for url: " + response.url, response=response
    )

    with patch("graderbot.answer_reader.requests.post", return_value=response):
        with pytest.raises(RuntimeError) as exc_info:
            reader.read(image, BOX)

    assert "super-secret-key" not in str(exc_info.value)


def test_google_vision_answer_reader_surfaces_the_error_body_on_a_bad_request(monkeypatch):
    """The default HTTPError message from raise_for_status() is just "400
    Client Error: Bad Request for url: ..." -- Google always puts the real
    reason (bad base64, payload too large, a missing field, ...) in the
    response body, so a caller shouldn't have to go digging for it."""
    reader = GoogleVisionAnswerReader(api_key="test-key")
    image = np.full((200, 200, 3), 255, np.uint8)
    response = MagicMock()
    response.url = "https://vision.googleapis.com/v1/images:annotate"
    response.text = (
        '{"error": {"code": 400, "message": '
        '"Request payload size exceeds the limit: 41943040 bytes.", '
        '"status": "INVALID_ARGUMENT"}}'
    )
    response.json.return_value = {
        "error": {
            "code": 400,
            "message": "Request payload size exceeds the limit: 41943040 bytes.",
            "status": "INVALID_ARGUMENT",
        }
    }
    response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "400 Client Error: Bad Request for url: " + response.url, response=response
    )

    with patch("graderbot.answer_reader.requests.post", return_value=response):
        with pytest.raises(RuntimeError, match="payload size exceeds the limit"):
            reader.read(image, BOX)


# -- Fraction detection (issue #70's "EasyOCR + OpenCV" combo) --------------


def _bar_image(bar_row_fraction=0.5, bar_length_fraction=0.9, width=200, height=100):
    """A synthetic answer-box crop with a real horizontal ink stroke drawn
    across it -- stands in for a handwritten fraction bar (or a decoy) so
    `_detect_fraction_bar`'s actual OpenCV line detection runs for real."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    y = int(height * bar_row_fraction)
    x_margin = int(width * (1 - bar_length_fraction) / 2)
    cv2.line(image, (x_margin, y), (width - x_margin, y), (0, 0, 0), 2)
    return image


def test_detect_fraction_bar_finds_a_long_central_horizontal_line():
    image = _bar_image(bar_row_fraction=0.5, bar_length_fraction=0.9, height=100)

    bar_y = _detect_fraction_bar(image)

    assert bar_y is not None
    assert abs(bar_y - 50) <= 5


def test_detect_fraction_bar_returns_none_for_a_blank_image():
    image = np.full((100, 200, 3), 255, dtype=np.uint8)

    assert _detect_fraction_bar(image) is None


def test_detect_fraction_bar_ignores_a_short_stroke():
    # Well under the 50%-of-width minimum -- e.g. a single digit's crossbar,
    # not a bar spanning the whole answer.
    image = _bar_image(bar_length_fraction=0.25)

    assert _detect_fraction_bar(image) is None


def _mock_ocr_response_sequence(*text_confidence_pairs):
    responses = []
    for text, confidence in text_confidence_pairs:
        responses.append(_mock_ocr_response(text, confidence))
    return responses


def test_easyocr_answer_reader_reads_a_detected_fraction():
    reader = EasyOcrAnswerReader(detect_fractions=True, service_url="http://localhost:8080")
    cropped = _bar_image(bar_row_fraction=0.5, bar_length_fraction=0.9, height=100)

    with patch(
        "graderbot.answer_reader.requests.post",
        side_effect=_mock_ocr_response_sequence(("3", 0.9), ("4", 0.8)),
    ) as mock_post:
        result = reader._read_as_fraction(cropped)

    assert result == OcrResult(
        text=r"\frac{3}{4}", raw_text=r"\frac{3}{4}", confidence=0.8, source=EASYOCR_SOURCE
    )
    # Each half is read with the narrow digit-only allowlist, regardless of
    # the reader's own `allowlist`.
    assert mock_post.call_args_list[0].kwargs["json"]["allowlist"] == "0123456789-"
    assert mock_post.call_args_list[1].kwargs["json"]["allowlist"] == "0123456789-"


def test_easyocr_answer_reader_falls_back_when_no_bar_is_found():
    reader = EasyOcrAnswerReader(detect_fractions=True, service_url="http://localhost:8080")
    cropped = np.full((100, 200, 3), 255, dtype=np.uint8)  # no bar in this crop

    assert reader._read_as_fraction(cropped) is None


def test_easyocr_answer_reader_falls_back_when_a_bar_is_too_close_to_an_edge():
    reader = EasyOcrAnswerReader(detect_fractions=True, service_url="http://localhost:8080")
    # A bar right at the top edge is more plausibly a stray mark/box border
    # than a fraction bar sitting between a numerator and a denominator.
    cropped = _bar_image(bar_row_fraction=0.02, bar_length_fraction=0.9, height=100)

    assert reader._read_as_fraction(cropped) is None


def test_easyocr_answer_reader_falls_back_when_a_half_reads_as_nothing():
    reader = EasyOcrAnswerReader(detect_fractions=True, service_url="http://localhost:8080")
    cropped = _bar_image(bar_row_fraction=0.5, bar_length_fraction=0.9, height=100)

    with patch(
        "graderbot.answer_reader.requests.post",
        side_effect=_mock_ocr_response_sequence(("3", 0.9), ("", None)),
    ):
        assert reader._read_as_fraction(cropped) is None


def test_easyocr_answer_reader_read_uses_fraction_result_when_available(monkeypatch):
    reader = EasyOcrAnswerReader(detect_fractions=True, service_url="http://localhost:8080")
    sentinel = OcrResult(text=r"\frac{1}{2}", raw_text=r"\frac{1}{2}", confidence=0.7, source=EASYOCR_SOURCE)
    monkeypatch.setattr(reader, "_read_as_fraction", lambda cropped: sentinel)
    monkeypatch.setattr(
        reader, "_call_service", lambda *a, **k: pytest.fail("whole-box read should not run")
    )
    image = np.full((200, 200, 3), 255, np.uint8)

    assert reader.read(image, BOX) is sentinel


def test_easyocr_answer_reader_read_falls_back_to_whole_box_when_fraction_is_none(monkeypatch):
    reader = EasyOcrAnswerReader(detect_fractions=True, service_url="http://localhost:8080")
    monkeypatch.setattr(reader, "_read_as_fraction", lambda cropped: None)
    sentinel = OcrResult(text="7", raw_text="7", confidence=0.9, source=EASYOCR_SOURCE)
    monkeypatch.setattr(reader, "_call_service", lambda image, allowlist: sentinel)
    image = np.full((200, 200, 3), 255, np.uint8)

    assert reader.read(image, BOX) is sentinel


def test_easyocr_answer_reader_skips_fraction_detection_when_disabled(monkeypatch):
    """detect_fractions defaults to False -- the whole point of the opt-in
    (issue #70) is that a run with no fraction questions never risks a
    false-positive bar detection corrupting a plain-number read."""
    reader = EasyOcrAnswerReader(service_url="http://localhost:8080")
    monkeypatch.setattr(
        reader, "_read_as_fraction", lambda cropped: pytest.fail("must not be called")
    )
    sentinel = OcrResult(text="7", raw_text="7", confidence=0.9, source=EASYOCR_SOURCE)
    monkeypatch.setattr(reader, "_call_service", lambda image, allowlist: sentinel)
    image = np.full((200, 200, 3), 255, np.uint8)

    assert reader.read(image, BOX) is sentinel
