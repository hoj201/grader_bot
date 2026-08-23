"""A tiny HTTP wrapper around EasyOCR (issue #70), run as its own container.

Kept out of the main graderbot deploy image on purpose: EasyOCR's only real
dependency, torch, ships no wheel for Intel Mac and is heavy (GPU-oriented)
to bundle alongside a small Streamlit app. graderbot.answer_reader.EasyOcrAnswerReader
is an HTTP client for this service, exactly the way graderbot.ocr talks to
Mathpix -- see EASYOCR_SERVICE_URL in the README.

Run locally with `docker compose up easyocr` (see docker-compose.yml at the
repo root). Not wired into the fly.io deploy yet -- see the README note on a
future GPU host (e.g. Modal) for this and other torch-dependent services.
"""

import base64
import re
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Built lazily, on first request, not at import time -- so `uvicorn main:app`
# (and any health check hitting a route other than /ocr) doesn't pay
# EasyOCR's multi-second model-load cost, and the process starts even if
# model download is briefly unreachable until an /ocr call actually needs it.
_reader = None

_DATA_URI_PATTERN = re.compile(r"^data:image/[a-zA-Z+.-]+;base64,(?P<data>.+)$", re.DOTALL)


def _get_reader():
    global _reader
    if _reader is None:
        # Imported here, not at module scope, so importing this module (e.g.
        # for a route that never touches OCR) doesn't force torch to load.
        import easyocr

        # gpu=False: this container has no GPU, on fly.io or anywhere else it
        # runs today.
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


class OcrRequest(BaseModel):
    image: str  # data URI, e.g. "data:image/png;base64,...."
    allowlist: str = ""


class OcrResponse(BaseModel):
    text: str
    confidence: Optional[float] = None


def _decode_image(data_uri: str) -> np.ndarray:
    match = _DATA_URI_PATTERN.match(data_uri)
    if not match:
        raise HTTPException(status_code=400, detail="image must be a data:image/...;base64,... URI")
    raw = base64.b64decode(match.group("data"))
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="could not decode image data")
    return image


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr", response_model=OcrResponse)
def ocr(request: OcrRequest) -> OcrResponse:
    image = _decode_image(request.image)
    reader = _get_reader()
    detections = reader.readtext(image, allowlist=request.allowlist or None)
    if not detections:
        return OcrResponse(text="", confidence=None)

    text = "".join(detected_text for _, detected_text, _ in detections)
    # The weakest detection sets the confidence for the whole read -- one
    # bad character shouldn't be hidden by an average across a clean rest.
    confidence = min(conf for _, _, conf in detections)
    return OcrResponse(text=text, confidence=float(confidence))
