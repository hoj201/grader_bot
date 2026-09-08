"""Tests for the AnswerReader strategies (issue #70): MathpixAnswerReader
(a thin wrapper over ocr.read_box), GoogleVisionAnswerReader (a direct REST
call to the Vision API), and NoOcrAnswerReader (issue #83). Mathpix/Vision
are mocked at the request boundary -- no real Mathpix/Vision API call is
needed.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from graderbot import answer_reader
from graderbot.answer_reader import (
    GOOGLE_VISION_SOURCE,
    NO_OCR_SOURCE,
    GoogleVisionAnswerReader,
    MathpixAnswerReader,
    NoOcrAnswerReader,
)
from graderbot.models import Box
from graderbot.ocr import OcrResult

BOX = Box(x_lower_left=0.1, y_lower_left=0.1, width=0.3, height=0.1)


def test_mathpix_answer_reader_delegates_to_read_box(monkeypatch):
    sentinel = OcrResult(text="12", raw_text="12", confidence=0.9)
    monkeypatch.setattr(answer_reader, "read_box", lambda image, box: sentinel)

    image = np.full((200, 200, 3), 255, np.uint8)
    assert MathpixAnswerReader().read(image, BOX) is sentinel


def test_no_ocr_answer_reader_never_reads_the_box(monkeypatch):
    """issue #83: NoOcrAnswerReader must not transcribe the crop at all --
    it always reports an empty response, regardless of what's in the box,
    and never makes a network call to do it."""
    monkeypatch.setattr(
        answer_reader, "requests", MagicMock(post=lambda *a, **k: pytest.fail("no network call expected"))
    )
    image = np.zeros((200, 200, 3), dtype=np.uint8)  # ink everywhere, not blank

    result = NoOcrAnswerReader().read(image, BOX)

    assert result == OcrResult(text="", raw_text="", confidence=None, source=NO_OCR_SOURCE)


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

