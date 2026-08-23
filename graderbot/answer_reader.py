"""Read the student's answer off a scanned worksheet's answer box (issue #70).

Three interchangeable strategies sit behind one `AnswerReader` protocol:

- `MathpixAnswerReader` -- wraps the existing `ocr.read_box`. This is what
  grading has always done, and remains the only choice that reads a
  handwritten LaTeX fraction (`\\frac{a}{b}`).
- `EasyOcrAnswerReader` -- calls a separate EasyOCR sidecar service (see
  `easyocr_service/main.py`) restricted to a small character allowlist, to
  work around Mathpix's college-level math OCR misreading sloppy handwritten
  digits (issue #70: "9" read as "G", "14" read as "1 h"). It cannot read
  fractions -- a worksheet with fraction answers should stay on Mathpix.

  The sidecar is a *separate service* rather than a graderbot dependency:
  EasyOCR's own dependency, torch, ships no wheel for Intel Mac and is heavy
  to bundle into the main deploy image. Two ways to run it -- see the README:
  `docker compose up -d easyocr` locally, or deployed on Modal
  (`easyocr_service/modal_app.py`) for anywhere the main app itself is
  hosted (e.g. fly.io) to reach.

  Optionally (`detect_fractions=True`) it also attempts a handwritten
  fraction: `_detect_fraction_bar` looks for a single long horizontal ink
  stroke spanning most of the box's width with OpenCV (geometrically
  distinct from any individual digit's much shorter strokes), splits the
  crop into numerator/denominator halves at that row, and OCRs each half
  separately. This is opt-in and off by default -- a false-positive bar
  detection on ordinary sloppy handwriting (the exact problem domain here)
  would silently corrupt a plain-number read, so a run that has no fraction
  questions should leave it off.
- `GoogleVisionAnswerReader` -- calls the Google Cloud Vision REST API
  (`DOCUMENT_TEXT_DETECTION`, Google's mode for dense/handwritten text)
  directly over `requests`, with no new project dependency and no sidecar --
  unlike EasyOCR, the Vision REST API is a plain HTTPS call, so it's as
  lightweight as Mathpix. It also has no math-symbol understanding and no
  character allowlist, so it cannot read fractions either.

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

from graderbot.imaging import _crop_box
from graderbot.models import Box
from graderbot.ocr import _BOX_INSET, OcrResult, read_box

EASYOCR_SOURCE = "easyocr"
GOOGLE_VISION_SOURCE = "google_vision"

# Answers are plain numbers, so the default stays narrow; broaden it per
# grading run from the Grade tab (e.g. append "xy" once algebra worksheets
# show up) instead of widening this default and reintroducing the kind of
# digit/letter confusion Mathpix already has.
EASYOCR_DEFAULT_ALLOWLIST = "0123456789."

_EASYOCR_SERVICE_URL_ENV = "EASYOCR_SERVICE_URL"
# Only required for the Modal deployment (see easyocr_service/modal_app.py);
# the local docker-compose service never checks this header, so leaving it
# unset is fine when EASYOCR_SERVICE_URL points at localhost.
_EASYOCR_API_KEY_ENV = "EASYOCR_API_KEY"

# A fraction's numerator/denominator are always plain (possibly negative)
# integers -- see grading._ANSWER_FRAC_PATTERN -- so each half is read with
# this narrower allowlist regardless of what `allowlist` is set to.
_FRACTION_PART_ALLOWLIST = "0123456789-"

# The bar must span at least this fraction of the box's width to count --
# long enough to rule out an individual digit's stroke (e.g. the crossbar of
# a "7" or "4"), which only spans a fraction of a multi-character answer's
# width, not the whole box.
_FRACTION_BAR_MIN_WIDTH_FRACTION = 0.5
# Near-horizontal tolerance, in pixels of vertical drift over the line's run.
_FRACTION_BAR_MAX_SLOPE_PX = 4
# The bar must sit at least this far from the top/bottom edges (as a
# fraction of the box's height) to be plausible -- a line hugging an edge is
# more likely a stray mark or the box border than a fraction bar sitting
# between a numerator and a denominator.
_FRACTION_BAR_EDGE_MARGIN_FRACTION = 0.15
# Thin band excluded around the detected bar itself when splitting, so the
# stroke's own pixels don't bleed into either half's OCR crop.
_FRACTION_BAR_SPLIT_MARGIN_FRACTION = 0.06

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


class EasyOcrAnswerReader:
    """Calls the `easyocr_service` sidecar (see module docstring), restricted
    to `allowlist`.

    Requires `EASYOCR_SERVICE_URL` (e.g. `http://localhost:8080` when running
    `docker compose up -d easyocr` locally, or the `*.modal.run` URL printed
    by `modal deploy easyocr_service/modal_app.py`) -- raises
    `EnvironmentError` if neither `service_url` nor the env var is set, the
    same failure mode `ocr._mathpix_ocr` uses for missing Mathpix
    credentials. `EASYOCR_API_KEY` is optional -- only the Modal deployment
    checks it (see `easyocr_service/main.py`); the local docker-compose
    service ignores it entirely.
    """

    def __init__(
        self,
        allowlist: str = EASYOCR_DEFAULT_ALLOWLIST,
        service_url: Optional[str] = None,
        detect_fractions: bool = False,
        api_key: Optional[str] = None,
    ):
        self.allowlist = allowlist
        self.detect_fractions = detect_fractions
        self.service_url = service_url or os.environ.get(_EASYOCR_SERVICE_URL_ENV)
        self.api_key = api_key or os.environ.get(_EASYOCR_API_KEY_ENV)
        if not self.service_url:
            raise EnvironmentError(
                f"{_EASYOCR_SERVICE_URL_ENV} must be set (e.g. in a .env file, "
                "pointing at `docker compose up -d easyocr` or a Modal deployment) "
                "to use EasyOcrAnswerReader"
            )

    def read(self, image: np.ndarray, box: Box) -> OcrResult:
        cropped = _crop_box(image, box, _BOX_INSET)
        if self.detect_fractions:
            fraction_result = self._read_as_fraction(cropped)
            if fraction_result is not None:
                return fraction_result
        return self._call_service(cropped, self.allowlist)

    def _read_as_fraction(self, cropped: np.ndarray) -> Optional[OcrResult]:
        """Attempts the numerator/denominator split described in the class
        docstring. Returns `None` (meaning: fall back to a normal whole-box
        read) whenever the split doesn't look trustworthy -- no bar found,
        the bar sits implausibly close to an edge, or either half reads as
        nothing -- rather than ever return a half-broken `\\frac{}{}`."""
        height = cropped.shape[0]
        bar_y = _detect_fraction_bar(cropped)
        if bar_y is None:
            return None
        edge_margin = _FRACTION_BAR_EDGE_MARGIN_FRACTION * height
        if not (edge_margin <= bar_y <= height - edge_margin):
            return None

        split_margin = max(1, int(_FRACTION_BAR_SPLIT_MARGIN_FRACTION * height))
        numerator_crop = cropped[: max(0, bar_y - split_margin), :]
        denominator_crop = cropped[bar_y + split_margin :, :]
        if numerator_crop.size == 0 or denominator_crop.size == 0:
            return None

        numerator = self._call_service(numerator_crop, _FRACTION_PART_ALLOWLIST)
        denominator = self._call_service(denominator_crop, _FRACTION_PART_ALLOWLIST)
        if not numerator.text or not denominator.text:
            return None

        text = f"\\frac{{{numerator.text}}}{{{denominator.text}}}"
        confidences = [c for c in (numerator.confidence, denominator.confidence) if c is not None]
        confidence = min(confidences) if confidences else None
        return OcrResult(text=text, raw_text=text, confidence=confidence, source=EASYOCR_SOURCE)

    def _call_service(self, image: np.ndarray, allowlist: str) -> OcrResult:
        success, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("Could not encode cropped box image")
        data_uri = "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

        headers = {"X-Api-Key": self.api_key} if self.api_key else {}
        response = requests.post(
            f"{self.service_url.rstrip('/')}/ocr",
            json={"image": data_uri, "allowlist": allowlist},
            headers=headers,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload.get("text", "")
        # No repair step exists for EasyOCR yet (unlike Mathpix's
        # _fix_stray_slashes/_fix_greek_misreads), so raw and final text are
        # the same value for now.
        return OcrResult(
            text=text, raw_text=text, confidence=payload.get("confidence"), source=EASYOCR_SOURCE
        )


def _detect_fraction_bar(image: np.ndarray) -> Optional[int]:
    """Looks for a single long, near-horizontal ink stroke spanning most of
    `image`'s width (an already-cropped answer box, RGB), via OpenCV's
    probabilistic Hough transform. Returns the stroke's row (pixel y
    coordinate), or `None` if nothing long/straight enough is found. Purely
    geometric -- it has no idea what the stroke actually is, which is why
    `EasyOcrAnswerReader._read_as_fraction` layers additional sanity checks
    (edge margin, both halves reading as something) on top of this."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    width = gray.shape[1]
    edges = cv2.Canny(gray, 50, 150)
    min_length = int(_FRACTION_BAR_MIN_WIDTH_FRACTION * width)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=40, minLineLength=min_length, maxLineGap=5
    )
    if lines is None:
        return None

    height = gray.shape[0]
    center = height / 2
    rows = [
        (y1 + y2) // 2
        for x1, y1, x2, y2 in lines.reshape(-1, 4)
        if abs(y2 - y1) <= _FRACTION_BAR_MAX_SLOPE_PX
    ]
    if not rows:
        return None
    # A genuine fraction bar sits between a numerator and a denominator, so
    # among candidate near-horizontal long strokes, the one closest to
    # vertical center is the most plausible -- an underline or a stray mark
    # is more likely to hug an edge instead.
    return min(rows, key=lambda y: abs(y - center))


def _min_word_confidence(full_text_annotation: dict) -> Optional[float]:
    """The lowest per-word confidence in a Vision `fullTextAnnotation`, or
    `None` if it contains no words -- same "weakest detection sets the
    confidence for the whole read" choice `easyocr_service` makes, so a
    single bad character isn't hidden by an average across a clean rest."""
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
        cropped = _crop_box(image, box, _BOX_INSET)
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
