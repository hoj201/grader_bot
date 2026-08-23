"""Read the student's answer off a scanned worksheet's answer box (issue #70).

Two interchangeable strategies sit behind one `AnswerReader` protocol:

- `MathpixAnswerReader` -- wraps the existing `ocr.read_box`. This is what
  grading has always done, and remains the only choice that reads a
  handwritten LaTeX fraction (`\\frac{a}{b}`).
- `EasyOcrAnswerReader` -- calls a separate EasyOCR sidecar service (see
  `easyocr_service/main.py`) restricted to a small character allowlist, to
  work around Mathpix's college-level math OCR misreading sloppy handwritten
  digits (issue #70: "9" read as "G", "14" read as "1 h"). It cannot read
  fractions -- a worksheet with fraction answers should stay on Mathpix.

  The sidecar is a *separate container* rather than a graderbot dependency:
  EasyOCR's own dependency, torch, ships no wheel for Intel Mac and is heavy
  to bundle into the main deploy image. It isn't wired into the fly.io
  deploy yet -- see the README for running it locally via docker-compose,
  and issue #70 for a future GPU host (e.g. Modal) to actually deploy it.

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

# Answers are plain numbers, so the default stays narrow; broaden it per
# grading run from the Grade tab (e.g. append "xy" once algebra worksheets
# show up) instead of widening this default and reintroducing the kind of
# digit/letter confusion Mathpix already has.
EASYOCR_DEFAULT_ALLOWLIST = "0123456789."

_EASYOCR_SERVICE_URL_ENV = "EASYOCR_SERVICE_URL"


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
    `docker compose up -d easyocr` locally) -- raises `EnvironmentError` if
    neither `service_url` nor the env var is set, the same failure mode
    `ocr._mathpix_ocr` uses for missing Mathpix credentials.
    """

    def __init__(self, allowlist: str = EASYOCR_DEFAULT_ALLOWLIST, service_url: Optional[str] = None):
        self.allowlist = allowlist
        self.service_url = service_url or os.environ.get(_EASYOCR_SERVICE_URL_ENV)
        if not self.service_url:
            raise EnvironmentError(
                f"{_EASYOCR_SERVICE_URL_ENV} must be set (e.g. in a .env file, "
                "pointing at `docker compose up -d easyocr`) to use EasyOcrAnswerReader"
            )

    def read(self, image: np.ndarray, box: Box) -> OcrResult:
        cropped = _crop_box(image, box, _BOX_INSET)
        success, encoded = cv2.imencode(".png", cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("Could not encode cropped box image")
        data_uri = "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

        response = requests.post(
            f"{self.service_url.rstrip('/')}/ocr",
            json={"image": data_uri, "allowlist": self.allowlist},
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
