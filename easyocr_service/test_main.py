"""Tests for the easyocr_service FastAPI app. Not runnable from the main
graderbot venv -- fastapi/easyocr/torch are intentionally not project
dependencies there (see graderbot/answer_reader.py's module docstring), and
`testpaths = ["tests"]` in the root pyproject.toml keeps `poetry run pytest`
from even trying to collect this file. Run these inside the container instead:

    docker compose build easyocr
    docker compose run --rm easyocr pytest -q

The real EasyOCR reader is heavy to load and network-dependent (first run
downloads model weights), so `_get_reader` is monkeypatched to a stub in
every test here -- these check the HTTP contract (request/response shape,
allowlist plumbing, confidence aggregation), not EasyOCR's own accuracy.
"""

import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client():
    return TestClient(main.app)


def _data_uri(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image)
    assert success
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


class _StubReader:
    def __init__(self, detections):
        self.detections = detections
        self.seen_allowlist = None

    def readtext(self, image, allowlist=None):
        self.seen_allowlist = allowlist
        return self.detections


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ocr_returns_text_and_confidence(client, monkeypatch):
    stub = _StubReader([([], "14", 0.91)])
    monkeypatch.setattr(main, "_get_reader", lambda: stub)

    image = np.full((40, 100, 3), 255, np.uint8)
    response = client.post("/ocr", json={"image": _data_uri(image), "allowlist": "0123456789."})

    assert response.status_code == 200
    assert response.json() == {"text": "14", "confidence": 0.91}


def test_ocr_passes_the_allowlist_through(client, monkeypatch):
    stub = _StubReader([([], "9", 0.5)])
    monkeypatch.setattr(main, "_get_reader", lambda: stub)

    image = np.full((40, 100, 3), 255, np.uint8)
    client.post("/ocr", json={"image": _data_uri(image), "allowlist": "0123456789."})

    assert stub.seen_allowlist == "0123456789."


def test_ocr_confidence_is_the_weakest_detection(client, monkeypatch):
    # Two detections concatenate into one response string; confidence takes
    # the lower of the two rather than an average, since one bad character
    # should not be hidden by a clean rest.
    stub = _StubReader([([], "1", 0.95), ([], "4", 0.40)])
    monkeypatch.setattr(main, "_get_reader", lambda: stub)

    image = np.full((40, 100, 3), 255, np.uint8)
    response = client.post("/ocr", json={"image": _data_uri(image), "allowlist": ""})

    assert response.json() == {"text": "14", "confidence": 0.40}


def test_ocr_returns_empty_text_and_null_confidence_when_nothing_detected(client, monkeypatch):
    monkeypatch.setattr(main, "_get_reader", lambda: _StubReader([]))

    image = np.full((40, 100, 3), 255, np.uint8)
    response = client.post("/ocr", json={"image": _data_uri(image), "allowlist": ""})

    assert response.json() == {"text": "", "confidence": None}


def test_ocr_rejects_a_non_data_uri_image(client):
    response = client.post("/ocr", json={"image": "not-a-data-uri", "allowlist": ""})
    assert response.status_code == 400


def test_ocr_allows_requests_with_no_api_key_when_none_is_configured(client, monkeypatch):
    # Local docker-compose never sets EASYOCR_API_KEY -- the check must be a
    # no-op there, not a lockout.
    monkeypatch.delenv("EASYOCR_API_KEY", raising=False)
    monkeypatch.setattr(main, "_get_reader", lambda: _StubReader([([], "14", 0.9)]))

    image = np.full((40, 100, 3), 255, np.uint8)
    response = client.post("/ocr", json={"image": _data_uri(image), "allowlist": ""})

    assert response.status_code == 200


def test_ocr_rejects_missing_api_key_when_one_is_configured(client, monkeypatch):
    monkeypatch.setenv("EASYOCR_API_KEY", "secret-value")
    monkeypatch.setattr(main, "_get_reader", lambda: _StubReader([([], "14", 0.9)]))

    image = np.full((40, 100, 3), 255, np.uint8)
    response = client.post("/ocr", json={"image": _data_uri(image), "allowlist": ""})

    assert response.status_code == 401


def test_ocr_rejects_wrong_api_key_when_one_is_configured(client, monkeypatch):
    monkeypatch.setenv("EASYOCR_API_KEY", "secret-value")
    monkeypatch.setattr(main, "_get_reader", lambda: _StubReader([([], "14", 0.9)]))

    image = np.full((40, 100, 3), 255, np.uint8)
    response = client.post(
        "/ocr",
        json={"image": _data_uri(image), "allowlist": ""},
        headers={"X-Api-Key": "wrong-value"},
    )

    assert response.status_code == 401


def test_ocr_accepts_correct_api_key_when_one_is_configured(client, monkeypatch):
    monkeypatch.setenv("EASYOCR_API_KEY", "secret-value")
    monkeypatch.setattr(main, "_get_reader", lambda: _StubReader([([], "14", 0.9)]))

    image = np.full((40, 100, 3), 255, np.uint8)
    response = client.post(
        "/ocr",
        json={"image": _data_uri(image), "allowlist": ""},
        headers={"X-Api-Key": "secret-value"},
    )

    assert response.status_code == 200


def test_health_ignores_api_key_entirely(client, monkeypatch):
    # Modal's own health probes shouldn't need the secret -- only /ocr is
    # gated.
    monkeypatch.setenv("EASYOCR_API_KEY", "secret-value")

    response = client.get("/health")

    assert response.status_code == 200
