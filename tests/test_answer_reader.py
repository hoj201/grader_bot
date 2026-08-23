"""Tests for the two AnswerReader strategies (issue #70): MathpixAnswerReader
(a thin wrapper over ocr.read_box) and EasyOcrAnswerReader (an HTTP client for
the easyocr_service sidecar). Both are mocked at the request boundary --
neither the real Mathpix API nor the easyocr_service container is needed.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from graderbot import answer_reader
from graderbot.answer_reader import (
    EASYOCR_DEFAULT_ALLOWLIST,
    EASYOCR_SOURCE,
    EasyOcrAnswerReader,
    MathpixAnswerReader,
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
