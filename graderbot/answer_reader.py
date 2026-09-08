"""Read the student's answer off a scanned worksheet's answer box (issue #70).

Three interchangeable strategies sit behind one `AnswerReader` protocol:

- `MathpixAnswerReader` -- wraps the existing `ocr.read_box`. This is what
  grading has always done, and remains the only choice that reads a
  handwritten LaTeX fraction (`\\frac{a}{b}`).
- `GoogleVisionAnswerReader` -- calls the Google Cloud Vision REST API
  (`DOCUMENT_TEXT_DETECTION`, Google's mode for dense/handwritten text)
  directly over `requests`, with no new project dependency and no sidecar --
  it's as lightweight as Mathpix. It also has no math-symbol understanding
  and no character allowlist, so it cannot read fractions either.
- `NoOcrAnswerReader` (issue #83) -- doesn't attempt to read the box's
  contents at all. By the time `grading.grade_hw` calls any reader's
  `.read()`, its own ink-detection blank check
  (`imaging.is_blank`) has already decided the box isn't blank -- so this
  reader just reports an empty response, which grading treats as answered
  but wrong and `markup` annotates with the correct answer, same as a wrong
  OCR read. This gives a student real per-question feedback (which answers
  they got right) without OCR ever having to transcribe -- and potentially
  misread -- their handwriting, for a class where neither OCR backend above
  reads reliably enough to trust. It is the default (issue #82): EasyOCR,
  the previous fallback for sloppy handwriting, was removed for reading
  student handwriting too poorly to trust, and no other backend has proven
  reliable enough to earn the default in its place.

Which one grading uses is a runtime choice made in the Grade tab, because
which backend reads a given class's handwriting best isn't known ahead of
time -- same reasoning as the two `name_reader.NameReader`s.
"""

import base64
import os
from typing import Optional, Protocol

import cv2
import numpy as np
import requests

from graderbot.imaging import crop_box_content_aware
from graderbot.models import Box
from graderbot.ocr import _BOX_INSET, OcrResult, read_box

GOOGLE_VISION_SOURCE = "google_vision"
NO_OCR_SOURCE = "no_ocr"

_GOOGLE_VISION_URL = "https://vision.googleapis.com/v1/images:annotate"
_GOOGLE_VISION_API_KEY_ENV = "GOOGLE_VISION_API_KEY"


class AnswerReader(Protocol):
    def read(self, image: np.ndarray, box: Box) -> OcrResult:
        """Read the response inside `box` on `image` (an already-loaded RGB
        numpy array, e.g. from `load_image_rgb`)."""
        ...


class MathpixAnswerReader:
    """The pre-issue-#70 default: Mathpix via `ocr.read_box`."""

    def read(self, image: np.ndarray, box: Box) -> OcrResult:
        return read_box(image, box)


class NoOcrAnswerReader:
    """Skips OCR entirely (issue #83) -- see the module docstring for why.

    `grade_hw` only ever calls a reader's `.read()` after its own blank
    check has already found ink in the box, so this reader doesn't need to
    look at `image`/`box` at all: it just reports "nothing legible read",
    which grading scores as wrong (an empty response never parses as a
    matching answer) and `markup` annotates with the correct answer, exactly
    like a wrong OCR read would.
    """

    def read(self, image: np.ndarray, box: Box) -> OcrResult:
        return OcrResult(text="", raw_text="", confidence=None, source=NO_OCR_SOURCE)


def _min_word_confidence(full_text_annotation: dict) -> Optional[float]:
    """The lowest per-word confidence in a Vision `fullTextAnnotation`, or
    `None` if it contains no words -- "weakest detection sets the confidence
    for the whole read", so a single bad character isn't hidden by an
    average across a clean rest."""
    confidences = [
        word["confidence"]
        for page in full_text_annotation.get("pages", [])
        for block in page.get("blocks", [])
        for paragraph in block.get("paragraphs", [])
        for word in paragraph.get("words", [])
        if "confidence" in word
    ]
    return min(confidences) if confidences else None


class GoogleVisionAnswerReader:
    """Calls the Google Cloud Vision REST API directly (see module
    docstring) -- no SDK, no sidecar, just `requests` like Mathpix.

    Requires `GOOGLE_VISION_API_KEY` -- raises `EnvironmentError` if neither
    `api_key` nor the env var is set, the same failure mode `ocr._mathpix_ocr`
    uses for missing Mathpix credentials.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get(_GOOGLE_VISION_API_KEY_ENV)
        if not self.api_key:
            raise EnvironmentError(
                f"{_GOOGLE_VISION_API_KEY_ENV} must be set (e.g. in a .env file) "
                "to use GoogleVisionAnswerReader"
            )

    def read(self, image: np.ndarray, box: Box) -> OcrResult:
        cropped = crop_box_content_aware(image, box, fallback_inset=_BOX_INSET)
        success, encoded = cv2.imencode(".png", cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("Could not encode cropped box image")
        content = base64.b64encode(encoded.tobytes()).decode("ascii")

        response = requests.post(
            _GOOGLE_VISION_URL,
            # The key goes in a header, not `?key=...`: a URL query param
            # ends up embedded in requests' own HTTPError message (and
            # anything else that logs response.url), leaking the key into
            # tracebacks/logs. The header is silent on failure.
            headers={"X-goog-api-key": self.api_key},
            json={
                "requests": [
                    {
                        "image": {"content": content},
                        # Google's mode for dense/handwritten text, as opposed
                        # to TEXT_DETECTION (tuned for sparse text like signs).
                        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                    }
                ]
            },
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            # The default HTTPError message is just "400 Client Error: Bad
            # Request for url: ..." -- useless on its own. Google always puts
            # the actual reason (bad base64, payload too large, a missing
            # field, ...) in the response body, so surface that instead of
            # leaving a caller to go dig for it. Read it from the body, never
            # from `e`/`response.url` -- same key-leak concern the header-vs-
            # query-param choice above guards against.
            detail = response.text
            try:
                detail = response.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                pass
            raise RuntimeError(f"Google Vision request failed: {detail}") from e
        result = response.json()["responses"][0]
        if "error" in result:
            # The Vision API reports a per-image failure (e.g. bad image
            # data, quota) inside a 200 response rather than an HTTP error
            # status, so raise_for_status() above won't catch it.
            raise RuntimeError(f"Google Vision error: {result['error'].get('message', result['error'])}")

        full_text_annotation = result.get("fullTextAnnotation", {})
        text = full_text_annotation.get("text", "").strip()
        confidence = _min_word_confidence(full_text_annotation)
        return OcrResult(text=text, raw_text=text, confidence=confidence, source=GOOGLE_VISION_SOURCE)
